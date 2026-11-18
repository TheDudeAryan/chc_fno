"""Report disk space and quota on the cluster.  It only reads.

For each filesystem you work on this prints its type, size, used, available
and inodes, then the quota, which is the limit that actually stops a write.
Nothing is deleted, moved or changed.

    python hpc_storage.py                  here, $HOME, $SCRATCH, /tmp, ...
    python hpc_storage.py PATH ...         and these paths too
    python hpc_storage.py --du PATH ...    and `du -sh` of those paths
    python hpc_storage.py --du             `du -sh` of Work, Results, Archive

Run it from the project root.  `du` reads every file, which is slow on a
shared filesystem and unwelcome on a login node: use hpc_storage.sh for that.
The report needs Linux (/proc/mounts, statvfs); anywhere else it says so and
stops.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import subprocess
import sys

import fno_paths

ENV_VARS = ("HOME", "SCRATCH", "WORK", "PROJECT", "PROJECTDIR", "LUSTRE",
            "TMPDIR")
TOOL_DIRS = ("/usr/lpp/mmfs/bin", "/opt/beegfs/sbin")
TMP_DIR = "/tmp"
GPFS = ("gpfs", "mmfs")
UNITS = ("B", "KB", "MB", "GB", "TB", "PB")
NOT_LINUX = ("hpc_storage.py needs Linux (it reads /proc/mounts and calls "
             "statvfs), and this is {platform}.  Run it on the cluster, "
             "with sbatch hpc_storage.sh")


def unsupported(platform=None):
    """Why this report cannot run on ``platform``, or None if it can."""
    platform = sys.platform if platform is None else platform
    if platform == "linux":
        return None
    return NOT_LINUX.format(platform=platform)


def human(n):
    """1536 -> '1.5 KB'."""
    n = float(n)
    for unit in UNITS[:-1]:
        if round(abs(n), 1) < 1024:
            return f"{n:.0f} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} {UNITS[-1]}"


def run(cmd, timeout=30):
    """The output of ``cmd``, or a line saying why it did not run."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True,
                           errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"could not run {cmd[0]}: {exc}"
    return p.stdout.strip()


def find_tool(name):
    """``name`` on PATH, else in the usual off-PATH install folders."""
    found = shutil.which(name)
    if found:
        return found
    for d in TOOL_DIRS:
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def quota_like_tools():
    """Every command named *quota*, which finds a site's own wrapper."""
    names = set()
    for d in os.environ.get("PATH", "").split(os.pathsep) + list(TOOL_DIRS):
        try:
            names.update(n for n in os.listdir(d) if "quota" in n.lower())
        except OSError:
            pass
    return sorted(names)


def unescape(field):
    """A /proc/mounts field with its octal escapes undone: a backslash and
    040 stand for a space."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), field)


def parse_mounts(lines):
    """(mount point, type, device) for each line of /proc/mounts."""
    mounts = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            dev, mnt, kind = parts[:3]
            mounts.append((unescape(mnt), kind, unescape(dev)))
    return mounts


def read_mounts(source="/proc/mounts"):
    try:
        with open(source, encoding="utf-8", errors="replace") as fh:
            return parse_mounts(fh.read().splitlines())
    except OSError:
        return []


def mount_info(path, mounts):
    """(type, device) of the mount that holds ``path``: the longest mount
    point it lies under.  ('?', '') if none does."""
    best, info = "", ("?", "")
    for mnt, kind, dev in mounts:
        inside = path == mnt or path.startswith(mnt.rstrip("/") + "/")
        if inside and len(mnt) > len(best):
            best, info = mnt, (kind, dev)
    return info


def targets(extra, env=None, cwd=None, tmp=None):
    """(labels, path) for each distinct filesystem among the working folder,
    the usual environment variables, /tmp and ``extra``, so a mount shared
    by several of them is reported once."""
    env = os.environ if env is None else env
    tmp = TMP_DIR if tmp is None else tmp
    cands = [("cwd", cwd or os.getcwd())]
    cands += [(f"${v}", env[v]) for v in ENV_VARS if env.get(v)]
    cands += [(tmp, tmp)] + [(p, p) for p in extra]
    by_dev: dict[int, tuple[list[str], str]] = {}
    for label, p in cands:
        p = os.path.realpath(p)
        if os.path.isdir(p):
            labels, _ = by_dev.setdefault(os.stat(p).st_dev, ([], p))
            labels.append(label)
    return list(by_dev.values())


def fs_numbers(path):
    """(size, used, available, inodes, inodes used) of the filesystem at
    ``path``, in bytes.  Available is what a user other than root can still
    write."""
    if sys.platform != "linux":
        raise OSError(NOT_LINUX.format(platform=sys.platform))
    s = os.statvfs(path)
    return (s.f_blocks * s.f_frsize,
            (s.f_blocks - s.f_bfree) * s.f_frsize,
            s.f_bavail * s.f_frsize,
            s.f_files, s.f_files - s.f_ffree)


def fs_lines(labels, path, kind, numbers):
    total, used, avail, files, files_used = numbers
    pct = 100.0 * used / total if total else 0.0
    out = [f"{' '.join(labels)}  ->  {path}  [{kind}]",
           f"  size {human(total)} | used {human(used)} ({pct:.0f}%) "
           f"| available {human(avail)}"]
    if files:
        out.append(f"  inodes: {files_used:,} used of {files:,}")
    return out


def report_fs(labels, path, mounts):
    kind = mount_info(path, mounts)[0]
    try:
        numbers = fs_numbers(path)
    except OSError as exc:
        print(f"{' '.join(labels)}  ->  {path}  [{kind}]\n"
              f"  cannot read the filesystem: {exc}")
        return
    print("\n".join(fs_lines(labels, path, kind, numbers)))


def group_names(path):
    """Your primary group and the group that owns ``path``: a group quota
    is kept per group."""
    if sys.platform != "linux":
        raise OSError(NOT_LINUX.format(platform=sys.platform))
    import grp
    names = []
    for gid in (os.getgid(), os.stat(path).st_gid):
        try:
            name = grp.getgrgid(gid).gr_name
        except KeyError:
            continue
        if name not in names:
            names.append(name)
    return names


def gpfs_fileset(path):
    """The GPFS fileset holding ``path``; home and scratch quotas are often
    set per fileset."""
    mmlsattr = find_tool("mmlsattr")
    if mmlsattr:
        for line in run([mmlsattr, "-L", path]).splitlines():
            if line.strip().startswith("fileset name:"):
                return line.split(":", 1)[1].strip()
    return None


def quota_plan(kind, dev, path, user, groups, fileset, find):
    """([commands], missing tool) to read the quota on a filesystem of this
    type.  A type with no known tool gives ([], None)."""
    if kind in GPFS:
        tool = find("mmlsquota")
        if tool is None:
            return [], "mmlsquota"
        dev = dev.split("/")[-1]

        def mmq(who, name):
            return [tool, who, name, "--block-size", "auto", dev]

        cmds = [mmq("-u", user)] + [mmq("-g", g) for g in groups]
        if fileset:
            cmds.append(mmq("-j", fileset))
        return cmds, None
    if kind == "lustre":
        name, args = "lfs", ["quota", "-h", "-u", user, path]
    elif kind == "beegfs":
        name, args = "beegfs-ctl", ["--getquota", "--uid", user]
    elif kind.startswith("nfs"):
        name, args = "quota", ["-s"]
    else:
        return [], None
    tool = find(name)
    if tool is None:
        return [], name
    return [[tool] + args], None


def report_quotas(paths, mounts):
    """What you may use.  The tools' own output, as their formats differ
    from site to site."""
    user = getpass.getuser()
    shown, missing = set(), []
    print(f"\n=== Quota: what you are allowed to use (user {user}) ===")
    for p in paths:
        kind, dev = mount_info(p, mounts)
        gpfs = kind in GPFS
        cmds, absent = quota_plan(
            kind, dev, p, user, group_names(p) if gpfs else [],
            gpfs_fileset(p) if gpfs else None, find_tool)
        if absent:
            missing.append((p, kind, absent))
        for cmd in cmds:
            if tuple(cmd) not in shown:
                shown.add(tuple(cmd))
                print("\n$ " + " ".join(cmd))
                print(run(cmd))
    for p, kind, tool in missing:
        print(f"\nNo `{tool}` found for {p} ({kind}): cannot read the "
              f"quota there.")
    tools = quota_like_tools()
    print("\nQuota-related commands visible here: "
          + (", ".join(tools) or "none"))
    print("If none of the above shows a limit, ask your HPC admins or "
          "check the site docs.")


def du_default():
    """The folders --du sizes when it is given none."""
    return [fno_paths.WORK, fno_paths.RESULTS, fno_paths.ARCHIVE]


def report_du(paths):
    for p in paths:
        if not os.path.exists(p):
            print(f"\n{p}: not there, nothing to size")
            continue
        real = os.path.realpath(p)
        print(f"\n$ du -sh {real}")
        print(run(["du", "-sh", real], timeout=600))


def main(argv=None):
    why = unsupported()
    if why:
        sys.exit(why)
    ap = argparse.ArgumentParser(
        description="Report disk space and quota.  Reads only; deletes "
                    "nothing.")
    ap.add_argument("paths", nargs="*", metavar="PATH",
                    help="more folders to report on")
    ap.add_argument("--du", action="store_true",
                    help="also size the PATHs with du -sh; with none, size "
                         "Work, Results and Archive")
    args = ap.parse_args(argv)
    extra = args.paths or (du_default() if args.du else [])
    found = targets(extra)
    mounts = read_mounts()
    for labels, path in found:
        report_fs(labels, path, mounts)
    report_quotas([p for _, p in found], mounts)
    if args.du:
        report_du(extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
