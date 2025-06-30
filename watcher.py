import os
import sys
import logging
from datetime import datetime
from inotify_simple import INotify, flags
#from pydbus import SessionBus

# --- Logging Setup ---
logging.basicConfig(
    filename="file_access.log",
    level=logging.INFO,
    format="%(asctime)s | PID: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

#def send_notification(title, message):
#    try:
#        bus = SessionBus()
#        notify = bus.get("org.freedesktop.Notifications", "/org/freedesktop/Notifications")
#        notify.Notify("FileWatcher", 0, "", title, message, [], {}, 5000)
#    except Exception as e:
#        print(f"❌ Notification failed: {e}")
def send_notification(title, message):
    print(f'[{title}] {message}')

# --- PID Resolver ---
def find_accessing_pid(target_path):
    for pid in filter(str.isdigit, os.listdir('/proc')):
        fd_dir = f'/proc/{pid}/fd'
        try:
            for fd in os.listdir(fd_dir):
                full_path = os.path.realpath(os.path.join(fd_dir, fd))
                if full_path == target_path:
                    return pid
        except (PermissionError, FileNotFoundError):
            continue
    return None

# --- Main Logic ---
def main():
    if len(sys.argv) != 2:
        print("Usage: sudo python watcher.py <path_to_file>")
        sys.exit(1)

    watch_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(watch_path):
        print(f"Error: {watch_path} is not a valid file.")
        sys.exit(1)

    inotify = INotify()
    wd = inotify.add_watch(watch_path, flags.ACCESS)

    print(f"👁️  Watching {watch_path} for access events (requires root)")
    while True:
        for event in inotify.read():
            if flags.ACCESS in flags.from_mask(event.mask):
                pid = find_accessing_pid(watch_path) or "Unknown"
                timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')
                message = f"File accessed by PID {pid} at {timestamp}"
                print(f"📣 {message}")
                logging.info(f"{pid} | {timestamp}")
                send_notification("📂 File Access Detected", message)

if __name__ == "__main__":
    main()
