import os
import psutil
import subprocess
import shutil
from typing import List, Dict, Optional, Union, Tuple, Set, Any
from pathlib import Path
from datetime import datetime, timedelta
import winreg
from logging_config import logger
from error_recovery import error_handler
import win32com.client
import json
import threading
from functools import wraps
import re
import stat

def require_elevation(func):
    """Decorator to check if function needs elevation."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PermissionError:
            logger.error(f"Operation requires elevation: {func.__name__}")
            return False, "This operation requires administrator privileges"
    return wrapper

def sanitize_path(path: str) -> str:
    """Sanitize file path to prevent directory traversal attacks."""
    return str(Path(path).resolve())

def is_safe_path(path: str, base_dir: str) -> bool:
    """Check if a path is safe (within base directory)."""
    try:
        resolved_path = Path(path).resolve()
        base_path = Path(base_dir).resolve()
        return base_path in resolved_path.parents
    except Exception:
        return False

class SystemController:
    def __init__(self):
        self._lock = threading.RLock()
        self.running_processes: Dict[str, int] = {}
        self.watched_processes: Set[int] = set()
        
        # Initialize common locations with proper path handling
        self.common_locations = {
            'desktop': str(Path.home() / 'Desktop'),
            'downloads': str(Path.home() / 'Downloads'),
            'documents': str(Path.home() / 'Documents'),
            'pictures': str(Path.home() / 'Pictures'),
            'music': str(Path.home() / 'Music'),
            'videos': str(Path.home() / 'Videos')
        }
        
        # Add common app paths with environment variable expansion
        self.common_apps = {
            'calculator': 'calc.exe',
            'notepad': 'notepad.exe',
            'chrome': os.path.expandvars(r'%ProgramFiles%\Google\Chrome\Application\chrome.exe'),
            'firefox': os.path.expandvars(r'%ProgramFiles%\Mozilla Firefox\firefox.exe'),
            'explorer': 'explorer.exe',
            'word': os.path.expandvars(r'%ProgramFiles%\Microsoft Office\root\Office16\WINWORD.EXE'),
            'excel': os.path.expandvars(r'%ProgramFiles%\Microsoft Office\root\Office16\EXCEL.EXE')
        }
        
        self.current_directory = str(Path.home())
        self.registered_apps = self._get_installed_apps()
        self.registered_apps.update(self.common_apps)
        
        # Start process monitoring
        self._start_process_monitor()

    def _start_process_monitor(self):
        """Start background process monitoring thread."""
        def monitor_processes():
            while True:
                try:
                    with self._lock:
                        for pid in list(self.watched_processes):
                            if not psutil.pid_exists(pid):
                                self.watched_processes.remove(pid)
                                # Remove from running_processes
                                for app, app_pid in list(self.running_processes.items()):
                                    if app_pid == pid:
                                        del self.running_processes[app]
                    threading.Event().wait(1.0)  # Check every second
                except Exception as e:
                    logger.error(f"Error in process monitor: {e}")
                    threading.Event().wait(5.0)  # Back off on error

        self._monitor_thread = threading.Thread(
            target=monitor_processes,
            daemon=True,
            name="ProcessMonitor"
        )
        self._monitor_thread.start()

    def _get_installed_apps(self) -> Dict[str, str]:
        """Get dictionary of installed applications and their paths with improved error handling."""
        apps = {}
        try:
            # Check Start Menu
            start_menu = Path.home() / 'AppData' / 'Roaming' / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs'
            shell = win32com.client.Dispatch("WScript.Shell")
            
            if start_menu.exists():
                for path in start_menu.rglob('*.lnk'):
                    try:
                        shortcut = shell.CreateShortCut(str(path))
                        target_path = shortcut.Targetpath
                        if os.path.exists(target_path) and target_path.lower().endswith('.exe'):
                            apps[path.stem.lower()] = target_path
                    except Exception as e:
                        logger.debug(f"Error processing shortcut {path}: {e}")

            # Check Program Files locations
            program_dirs = [
                os.environ.get('ProgramFiles', 'C:/Program Files'),
                os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')
            ]
            
            for program_dir in program_dirs:
                if os.path.exists(program_dir):
                    for root, _, files in os.walk(program_dir):
                        for file in files:
                            if file.lower().endswith('.exe'):
                                try:
                                    full_path = os.path.join(root, file)
                                    apps[Path(file).stem.lower()] = full_path
                                except Exception as e:
                                    logger.debug(f"Error processing exe {file}: {e}")

        except Exception as e:
            logger.error(f"Error getting installed apps: {e}")

        return apps

    @error_handler.with_retry("launch_application")
    def launch_application(self, app_name: str) -> str:
        """Launch application with dialogue-friendly response."""
        try:
            if not app_name:
                return "I need an application name to proceed."
                
            # Clean up app name
            app_name = app_name.lower().strip()
            if not app_name.endswith('.exe'):
                app_name += '.exe'
                
            # Security check for potentially dangerous applications
            if any(keyword in app_name.lower() for keyword in ['cmd', 'powershell', 'reg', 'taskmgr']):
                return "I apologize, but I cannot access system utilities for security reasons."
                
            # Find application path
            app_path = None
            if app_name in self.registered_apps:
                app_path = self.registered_apps[app_name]
            else:
                for path in self.registered_apps.values():
                    if path.lower().endswith(app_name):
                        app_path = path
                        break
            
            if not app_path or not os.path.exists(app_path):
                return f"I couldn't find {app_name}. Are you sure it's installed?"
                
            # Verify file is executable
            if not os.access(app_path, os.X_OK):
                return f"I don't have permission to launch {app_name}."
                
            # Launch process with proper security context
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            process = subprocess.Popen(
                app_path,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            
            with self._lock:
                self.running_processes[app_name] = process.pid
                self.watched_processes.add(process.pid)
                
            return f"I've launched {app_name} for you."
        
        except Exception as e:
            logger.error(f"Error launching {app_name}: {e}")
            return f"I encountered an error trying to launch {app_name}. {str(e)}"

    def close_application(self, app_name: str) -> Tuple[bool, str]:
        """Close an application with improved process handling and security."""
        try:
            if not app_name:
                return False, "No application name provided"
                
            app_name = app_name.lower().strip()
            if not app_name.endswith('.exe'):
                app_name += '.exe'
                
            success = False
            with self._lock:
                # First try to close our tracked process
                if app_name in self.running_processes:
                    pid = self.running_processes[app_name]
                    try:
                        process = psutil.Process(pid)
                        process.terminate()
                        success = process.wait(timeout=5) == 0
                        if success:
                            self.running_processes.pop(app_name, None)
                            self.watched_processes.discard(pid)
                            return True, f"Closed {app_name}"
                    except psutil.NoSuchProcess:
                        self.running_processes.pop(app_name, None)
                        self.watched_processes.discard(pid)
                    except Exception as e:
                        logger.error(f"Error closing process {pid}: {e}")
                
            # If not found in our tracked processes, try to find it in running processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'].lower() == app_name:
                        proc.terminate()
                        success = proc.wait(timeout=5) == 0
                        if success:
                            return True, f"Closed {app_name}"
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.debug(f"Error accessing process: {e}")
                    continue
                    
            return success, f"Could not close {app_name}"
            
        except Exception as e:
            logger.error(f"Error closing {app_name}: {e}")
            return False, f"Error closing {app_name}"

    def list_directory(self, path: Optional[str] = None) -> str:
        """List directory contents with conversation-friendly response."""
        try:
            contents = self._list_directory_contents(path)
            
            if not contents:
                if not path:
                    path = "current directory"
                return f"I couldn't find any files in the {path}."
            
            # Format the response in a conversational way
            files = [item for item in contents if item['type'] == 'file']
            folders = [item for item in contents if item['type'] == 'directory']
            
            response_parts = []
            if folders:
                folder_list = ", ".join(folder['name'] for folder in folders[:5])
                if len(folders) > 5:
                    folder_list += f" and {len(folders) - 5} more folders"
                response_parts.append(f"I found these folders: {folder_list}")
            
            if files:
                file_list = ", ".join(file['name'] for file in files[:5])
                if len(files) > 5:
                    file_list += f" and {len(files) - 5} more files"
                response_parts.append(f"Here are the files: {file_list}")
            
            return " ".join(response_parts)
            
            
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return "I encountered an error trying to list the directory contents."

    def navigate_to(self, location: str) -> str:
        """Navigate to directory with conversation-friendly response."""
        try:
            if not location:
                return "I need a location to navigate to."
                
            location = location.lower().strip()
            new_path = None
            
            # Check common locations
            if location in self.common_locations:
                new_path = self.common_locations[location]
            else:
                # Try to resolve path
                potential_path = Path(location)
                if potential_path.is_absolute():
                    new_path = str(potential_path)
                else:
                    new_path = str(Path(self.current_directory) / location)
            
            # Security checks
            new_path = sanitize_path(new_path)
            if not os.path.exists(new_path):
                return f"I couldn't find the location: {location}"
                
            if not os.path.isdir(new_path):
                return f"'{location}' is not a folder."
                
            if not os.access(new_path, os.R_OK):
                return f"I don't have permission to access {location}"
            
            self.current_directory = new_path
            return f"I've navigated to {location} for you."
                
        except Exception as e:
            logger.error(f"Error navigating to {location}: {e}")
            return f"I encountered an error trying to navigate to {location}."

    def get_system_info(self) -> str:
        """Get system information with conversation-friendly response."""
        try:
            info = self._get_system_info_raw()
            
            # Format the information conversationally
            response_parts = []
            
            # CPU info
            cpu_usage = info.get('cpu_percent', 0)
            response_parts.append(f"CPU usage is at {cpu_usage}%")
            
            # Memory info
            if 'memory_percent' in info:
                memory_usage = info['memory_percent']
                response_parts.append(f"Memory usage is at {memory_usage}%")
            
            # Battery info
            battery = info.get('battery', {})
            if battery and battery.get('percent') is not None:
                battery_status = (
                    f"Battery is at {battery['percent']}% and "
                    f"{'charging' if battery['power_plugged'] else 'discharging'}"
                )
                response_parts.append(battery_status)
            
            # Running applications
            running_apps = info.get('running_apps', [])
            if running_apps:
                app_list = ", ".join(running_apps[:3])
                if len(running_apps) > 3:
                    app_list += f" and {len(running_apps) - 3} other applications"
                response_parts.append(f"Currently running: {app_list}")
            
            return " | ".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return "I encountered an error trying to get system information."

    def _get_battery_info(self) -> Dict[str, Union[float, str]]:
        """Get battery information with improved error handling."""
        try:
            battery = psutil.sensors_battery()
            if battery:
                time_left = str(timedelta(seconds=battery.secsleft)) if battery.secsleft > 0 else 'Unknown'
                return {
                    'percent': battery.percent,
                    'power_plugged': battery.power_plugged,
                    'time_left': time_left,
                    'status': 'Charging' if battery.power_plugged else 'Discharging'
                }
        except Exception as e:
            logger.error(f"Error getting battery info: {e}")
        return {
            'percent': None,
            'power_plugged': None,
            'time_left': 'Unknown',
            'status': 'Unknown'
        }

    def search_files(self, query: str, path: Optional[str] = None) -> str:
        """Search files with conversation-friendly response."""
        try:
            if not query:
                return "I need something to search for."
            
            results = self._search_files_raw(query, path)
            
            if not results:
                location_str = f" in {path}" if path else ""
                return f"I couldn't find any files matching '{query}'{location_str}."
            
            # Format the response conversationally
            files = [item for item in results if item['type'] == 'file']
            folders = [item for item in results if item['type'] == 'directory']
            
            response_parts = []
            if folders:
                folder_list = ", ".join(folder['name'] for folder in folders[:3])
                if len(folders) > 3:
                    folder_list += f" and {len(folders) - 3} more folders"
                response_parts.append(f"I found these matching folders: {folder_list}")
            
            if files:
                file_list = ", ".join(file['name'] for file in files[:3])
                if len(files) > 3:
                    file_list += f" and {len(files) - 3} more files"
                response_parts.append(f"And these matching files: {file_list}")
            
            return " ".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return "I encountered an error while searching for files."

    def control_application(self, action: str, app_name: str) -> Tuple[bool, str]:
        """Control applications with improved action handling and security."""
        try:
            if not action or not app_name:
                return False, "Action and application name must be provided"

            action = action.lower().strip()
            valid_actions = {
                'close': self.close_application,
                'open': self.launch_application,
                'launch': self.launch_application,
                'start': self.launch_application
            }

            if action not in valid_actions:
                return False, f"Invalid action: {action}"

            return valid_actions[action](app_name)

        except Exception as e:
            logger.error(f"Error controlling application: {e}")
            return False, f"Error controlling {app_name}: {str(e)}"

    def get_running_processes(self) -> List[Dict[str, Union[str, float]]]:
        """Get list of running processes with improved detail and error handling."""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    with proc.oneshot():
                        pinfo = proc.as_dict(attrs=[
                            'pid', 'name', 'cpu_percent', 'memory_percent',
                            'status', 'create_time', 'username', 'cmdline'
                        ])
                        
                        processes.append({
                            'pid': pinfo['pid'],
                            'name': pinfo['name'],
                            'cpu_percent': pinfo['cpu_percent'] or 0.0,
                            'memory_percent': pinfo['memory_percent'] or 0.0,
                            'status': pinfo['status'],
                            'start_time': datetime.fromtimestamp(pinfo['create_time']).isoformat(),
                            'username': pinfo['username'],
                            'command': ' '.join(pinfo['cmdline']) if pinfo['cmdline'] else ''
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                except Exception as e:
                    logger.debug(f"Error getting process info: {e}")
                    continue

            return sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting processes: {e}")
            return []

    def get_running_applications(self) -> List[Dict[str, Any]]:
        """Get list of running applications with enhanced details."""
        try:
            running_apps = []
            monitored_pids = set(self.running_processes.values())
            
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    proc_info = proc.as_dict(attrs=['pid', 'name', 'exe'])
                    if proc_info['exe'] and proc_info['exe'].lower().endswith('.exe'):
                        is_monitored = proc_info['pid'] in monitored_pids
                        app_info = {
                            'name': proc_info['name'],
                            'path': proc_info['exe'],
                            'pid': proc_info['pid'],
                            'monitored': is_monitored,
                            'status': 'Running'
                        }
                        
                        if is_monitored:
                            app_info['launch_time'] = proc.create_time()
                            try:
                                app_info['cpu_usage'] = proc.cpu_percent(interval=0.1)
                                app_info['memory_usage'] = proc.memory_percent()
                            except Exception:
                                pass
                                
                        running_apps.append(app_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                except Exception as e:
                    logger.debug(f"Error getting application info: {e}")
                    continue
                    
            return sorted(running_apps, key=lambda x: x['name'].lower())
            
        except Exception as e:
            logger.error(f"Error getting running applications: {e}")
            return []

    def cleanup(self):
        """Cleanup system controller resources."""
        try:
            # Stop monitoring thread
            if hasattr(self, '_monitor_thread'):
                self.stop_event = threading.Event()
                self.stop_event.set()
                self._monitor_thread.join(timeout=5)

            # Clean up running processes
            with self._lock:
                for app_name, pid in list(self.running_processes.items()):
                    try:
                        if psutil.pid_exists(pid):
                            proc = psutil.Process(pid)
                            proc.terminate()
                            proc.wait(timeout=3)
                    except Exception as e:
                        logger.debug(f"Error cleaning up process {pid}: {e}")

                self.running_processes.clear()
                self.watched_processes.clear()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Create global instance
system_controller = SystemController()

# Register cleanup
import atexit
atexit.register(system_controller.cleanup)

if __name__ == "__main__":
    # Test system control functionality
    print("\nTesting System Controller...")
    
    # Test application launch
    success, message = system_controller.launch_application("notepad")
    print(f"Launch notepad: {message}")
    
    # Test directory listing
    print("\nListing desktop contents:")
    contents = system_controller.list_directory("desktop")
    print(json.dumps(contents, indent=2))
    
    # Test system info
    print("\nSystem Information:")
    info = system_controller.get_system_info()
    print(json.dumps(info, indent=2))
    
    # Test file search
    print("\nSearching for python files:")
    results = system_controller.search_files(".py")
    print(json.dumps(results, indent=2))
    
    # Cleanup
    system_controller.cleanup()