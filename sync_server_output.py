import socket
import sys
from pathlib import Path

import paramiko


HOST = "region-41.seetacloud.com"
PORT = 45423
USER = "root"
PASSWORD = "m0bXX1jW3rzP"

REMOTE_TAR = "/tmp/graphrag_output_server.tar.gz"
REMOTE_SRC = "/root/autodl-tmp/GraphRAG-master/GraphRAG-master/output"
LOCAL_TAR = Path(r"D:\Tools\Downloads\GraphRAG-master1\root\autodl-tmp\GraphRAG-master\GraphRAG-master\output\graphrag_output_server.tar.gz")
LOG_FILE = Path(r"D:\Tools\Downloads\GraphRAG-master1\root\autodl-tmp\GraphRAG-master\GraphRAG-master\output\sync_server_output.log")


def log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def main() -> int:
    log("sync_server_output: start")
    LOCAL_TAR.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL_TAR.exists():
        LOCAL_TAR.unlink()

    sock = socket.create_connection((HOST, PORT), timeout=30)
    trans = paramiko.Transport(sock)
    trans.banner_timeout = 60
    trans.auth_timeout = 60
    trans.start_client(timeout=60)
    trans.auth_password(USER, PASSWORD, fallback=False)
    if not trans.is_authenticated():
        raise RuntimeError("SSH authentication failed")

    log("sync_server_output: authenticated")

    chan = trans.open_session()
    chan.exec_command(f"tar -czf {REMOTE_TAR} -C /root/autodl-tmp/GraphRAG-master/GraphRAG-master output")
    exit_code = chan.recv_exit_status()
    stdout = chan.makefile("r", -1).read().decode("utf-8", errors="replace").strip()
    stderr = chan.makefile_stderr("r", -1).read().decode("utf-8", errors="replace").strip()
    chan.close()
    if exit_code != 0:
        raise RuntimeError(f"remote tar failed: exit={exit_code} stdout={stdout} stderr={stderr}")

    log("sync_server_output: remote tar ready")

    sftp = paramiko.SFTPClient.from_transport(trans)
    sftp.get(REMOTE_TAR, str(LOCAL_TAR))
    remote_size = sftp.stat(REMOTE_TAR).st_size
    sftp.close()

    log(f"sync_server_output: local_tar={LOCAL_TAR}")
    log(f"sync_server_output: remote_size={remote_size}")
    log(f"sync_server_output: local_size={LOCAL_TAR.stat().st_size}")

    chan2 = trans.open_session()
    chan2.exec_command(f"rm -f {REMOTE_TAR}")
    chan2.recv_exit_status()
    chan2.close()

    trans.close()
    log("sync_server_output: done")
    print(str(LOCAL_TAR))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"sync_server_output: error: {exc}\n")
        raise
