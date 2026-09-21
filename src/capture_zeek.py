#!/usr/bin/env python3
"""
Zeek Traffic Capture Script for MLOps-AdaptiveNIDS

Captures real-time network traffic on a specified interface using Zeek,
saving output log files to sequentially numbered directories in `zeek_logs/`.

Automatically adjusts file ownership so created directories and logs
are owned by the non-root user even when run with `sudo`.
"""

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def get_available_interfaces():
    """Retrieve list of available network interfaces on the system."""
    interfaces = []
    
    if hasattr(socket, 'if_nameindex'):
        try:
            interfaces = [name for _, name in socket.if_nameindex()]
        except Exception:
            pass
            
    if not interfaces and os.path.exists('/sys/class/net'):
        try:
            interfaces = os.listdir('/sys/class/net')
        except Exception:
            pass
            
    return interfaces


def find_zeek_binary():
    """Locate the zeek executable in PATH or standard installation directories."""
    binary = shutil.which('zeek')
    if binary:
        return binary
        
    common_paths = [
        '/opt/zeek/bin/zeek',
        '/usr/local/zeek/bin/zeek',
        '/usr/bin/zeek',
        '/usr/local/bin/zeek'
    ]
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
            
    return None


def fix_ownership(target_path: Path):
    """
    If running under sudo or as root, recursively change ownership of target_path
    and all its contents to the original non-root user (SUDO_UID/SUDO_GID).
    """
    if not hasattr(os, 'chown'):
        return

    target_uid = None
    target_gid = None

    if 'SUDO_UID' in os.environ and 'SUDO_GID' in os.environ:
        try:
            target_uid = int(os.environ['SUDO_UID'])
            target_gid = int(os.environ['SUDO_GID'])
        except ValueError:
            pass

    if target_uid is None:
        # Fallback to parent directory owner if not running via sudo env
        parent = target_path.parent if target_path.exists() else target_path
        if parent.exists():
            stat_info = parent.stat()
            target_uid = stat_info.st_uid
            target_gid = stat_info.st_gid

    if target_uid is not None and target_gid is not None:
        paths_to_fix = [target_path]
        if target_path.is_dir():
            try:
                paths_to_fix.extend(target_path.rglob('*'))
            except Exception:
                pass

        for p in paths_to_fix:
            try:
                os.chown(p, target_uid, target_gid)
            except Exception:
                pass


def get_next_numbered_dir(parent_dir: Path, prefix: str = "run_") -> Path:
    """Scan parent_dir and determine the next incremented directory path."""
    parent_dir.mkdir(parents=True, exist_ok=True)
    fix_ownership(parent_dir)
    
    existing_dirs = [
        d.name for d in parent_dir.iterdir()
        if d.is_dir() and (d.name.startswith(prefix) or d.name.isdigit())
    ]
    
    max_num = 0
    pattern = re.compile(rf"^(?:{re.escape(prefix)})?(\d+)$")
    
    for d_name in existing_dirs:
        match = pattern.match(d_name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                
    next_num = max_num + 1
    next_dir = parent_dir / f"{prefix}{next_num}"
    next_dir.mkdir(parents=True, exist_ok=True)
    fix_ownership(next_dir)
    return next_dir


def main():
    parser = argparse.ArgumentParser(
        description="Capture live network traffic with Zeek into numbered log directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--interface",
        type=str,
        help="Network interface to capture traffic from (e.g. eth0, wlo1, lo)"
    )
    parser.add_argument(
        "-l", "--list-interfaces",
        action="store_true",
        help="List available network interfaces on this machine and exit"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="zeek_logs",
        help="Parent directory to store output logs (default: zeek_logs)"
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        default="run_",
        help="Directory name prefix for numbered output folders (default: run_)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=None,
        help="Optional capture duration in seconds (default: capture until stopped manually)"
    )
    parser.add_argument(
        "-f", "--filter",
        type=str,
        default=None,
        help="BPF filter string to pass to Zeek (e.g. 'ip' or 'not ip6')"
    )
    parser.add_argument(
        "--ipv4-only", "--no-ipv6",
        action="store_true",
        dest="ipv4_only",
        help="Capture IPv4 packets only and ignore IPv6 (equivalent to -f 'ip')"
    )

    args = parser.parse_args()
    available_ifaces = get_available_interfaces()

    # Determine BPF filter
    bpf_filter = args.filter
    if args.ipv4_only and not bpf_filter:
        bpf_filter = "ip"

    # List interfaces request
    if args.list_interfaces:
        print("\n🌐 Available Network Interfaces:")
        print("=" * 35)
        for idx, iface in enumerate(available_ifaces, 1):
            print(f"  [{idx}] {iface}")
        print("=" * 35)
        sys.exit(0)

    # Validate interface parameter
    if not args.interface:
        print("❌ Error: Network interface not specified.")
        print("\n🌐 Available Network Interfaces:")
        for idx, iface in enumerate(available_ifaces, 1):
            print(f"  [{idx}] {iface}")
        print("\nUsage example:")
        print(f"  sudo python3 {sys.argv[0]} -i {available_ifaces[0] if available_ifaces else 'wlo1'}")
        sys.exit(1)

    if args.interface not in available_ifaces:
        print(f"⚠️  Warning: '{args.interface}' was not found in available system interfaces: {available_ifaces}")

    # Check for Zeek installation
    zeek_bin = find_zeek_binary()
    if not zeek_bin:
        print("❌ Error: Zeek binary could not be found.")
        print("Please ensure Zeek is installed and available in PATH or /opt/zeek/bin/zeek.")
        sys.exit(1)

    # Privilege check
    is_root = (os.geteuid() == 0) if hasattr(os, 'geteuid') else False
    if not is_root:
        print("\n⚠️  Notice: Running without root (sudo) privileges.")
        print("   Capturing raw network packets typically requires root / CAP_NET_RAW capabilities.")
        print(f"   If packet socket creation fails, run: sudo python3 {' '.join(sys.argv)}")
        print(f"   Or grant capabilities once:         sudo setcap cap_net_raw,cap_net_admin=eip {zeek_bin}\n")

    # Prepare output directory
    project_root = Path(__file__).resolve().parent.parent
    parent_output_dir = project_root / args.output_dir
    target_dir = get_next_numbered_dir(parent_output_dir, prefix=args.prefix)

    print("🛡️  Starting Zeek Network Traffic Capture")
    print("=" * 55)
    print(f"  • Interface      : {args.interface}")
    print(f"  • Zeek Binary    : {zeek_bin}")
    print(f"  • Log Directory  : {target_dir}")
    print(f"  • Privileges     : {'Root / Elevated' if is_root else 'Non-root (sudo may be needed)'}")
    if bpf_filter:
        print(f"  • BPF Filter     : {bpf_filter}")
    if args.duration:
        print(f"  • Duration       : {args.duration} seconds")
    else:
        print("  • Duration       : Continuous (Press Ctrl+C to stop)")
    print("=" * 55)

    cmd = [zeek_bin, "-i", args.interface]
    if bpf_filter:
        cmd.extend(["-f", bpf_filter])
    capture_success = True
    err_msg = ""
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(target_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"\n🚀 Zeek is now capturing on interface '{args.interface}'...")
        print(f"📁 Logs are being written to: {target_dir}\n")

        start_time = time.time()
        while True:
            ret = process.poll()
            if ret is not None:
                _, stderr = process.communicate()
                err_msg = stderr.decode('utf-8', errors='ignore')
                if ret != 0:
                    capture_success = False
                    print(f"❌ Zeek exited unexpectedly with error code {ret}")
                    if err_msg:
                        print(f"\nZeek Error Details:\n{err_msg.strip()}")
                break

            if args.duration and (time.time() - start_time) >= args.duration:
                print(f"\n⏰ Duration limit of {args.duration} seconds reached. Stopping capture...")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Stop signal received (Ctrl+C). Terminating Zeek capture gracefully...")
    finally:
        if 'process' in locals() and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        # Fix ownership of created directory and files so normal user can read/delete them
        fix_ownership(target_dir)

    if not capture_success:
        if "CAP_NET_RAW" in err_msg or "pcap_error" in err_msg:
            print("\n🔑 Root privileges required to open raw packet socket!")
            print(f"👉 Fix option 1 (Run with sudo):")
            print(f"   sudo python3 {' '.join(sys.argv)}")
            print(f"\n👉 Fix option 2 (Grant cap_net_raw permission once):")
            print(f"   sudo setcap cap_net_raw,cap_net_admin=eip {zeek_bin}\n")
        sys.exit(1)

    # Symlink / copy latest conn.log for convenience if it exists
    conn_log_path = target_dir / "conn.log"
    if conn_log_path.exists():
        root_conn_log = parent_output_dir / "conn.log"
        try:
            if root_conn_log.is_symlink() or root_conn_log.exists():
                root_conn_log.unlink()
            shutil.copyfile(conn_log_path, root_conn_log)
            fix_ownership(root_conn_log)
            print(f"✅ Updated primary log: {root_conn_log}")
        except Exception:
            pass

    # Final sweep to ensure user ownership of the parent zeek_logs directory
    fix_ownership(parent_output_dir)

    print(f"✅ Capture completed successfully. Logs saved in {target_dir}\n")


if __name__ == "__main__":
    main()
