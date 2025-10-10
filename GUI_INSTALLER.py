"""
FaB Card Detector - GUI Installer
Automatic installation with progress bar and optional debug mode
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import sys
import os
import platform
import urllib.request
import threading
from pathlib import Path

class InstallerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FaB Card Detector - Installer")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Configuration
        self.required_python_major = 3
        self.required_python_minor = 8
        self.python_download_version = "3.11.9"
        self.python_installer_url = f"https://www.python.org/ftp/python/{self.python_download_version}/python-{self.python_download_version}-amd64.exe"
        
        self.show_console = tk.BooleanVar(value=False)
        self.installation_complete = False
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Header
        header = tk.Label(
            self.root, 
            text="FaB Card Detector Installer", 
            font=("Arial", 18, "bold"),
            pady=20
        )
        header.pack()
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready to install",
            font=("Arial", 10),
            fg="blue"
        )
        self.status_label.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            mode='indeterminate',
            length=500
        )
        self.progress.pack(pady=10)
        
        # Console output (hidden by default)
        self.console_frame = tk.Frame(self.root)
        self.console_text = scrolledtext.ScrolledText(
            self.console_frame,
            height=15,
            width=70,
            font=("Consolas", 9)
        )
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        # Show console checkbox
        show_console_check = tk.Checkbutton(
            self.root,
            text="Show detailed output (for troubleshooting)",
            variable=self.show_console,
            command=self._toggle_console
        )
        show_console_check.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        self.install_button = tk.Button(
            button_frame,
            text="Install",
            command=self._start_installation,
            width=15,
            height=2,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.install_button.grid(row=0, column=0, padx=10)
        
        self.close_button = tk.Button(
            button_frame,
            text="Close",
            command=self.root.quit,
            width=15,
            height=2,
            font=("Arial", 10)
        )
        self.close_button.grid(row=0, column=1, padx=10)
        self.close_button.config(state=tk.DISABLED)
        
    def _toggle_console(self):
        if self.show_console.get():
            self.console_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
            self.root.geometry("600x700")
        else:
            self.console_frame.pack_forget()
            self.root.geometry("600x500")
    
    def _log(self, message, color="black"):
        """Add message to console and status"""
        self.console_text.insert(tk.END, message + "\n")
        self.console_text.see(tk.END)
        self.root.update()
        
        # Update status label (just the last line)
        if "\n" not in message:
            self.status_label.config(text=message)
    
    def _start_installation(self):
        """Start installation in a separate thread"""
        self.install_button.config(state=tk.DISABLED)
        self.progress.start(10)
        
        # Run installation in thread to keep GUI responsive
        thread = threading.Thread(target=self._run_installation)
        thread.daemon = True
        thread.start()
    
    def _run_installation(self):
        """Main installation logic"""
        try:
            # Check Python
            self._log("Checking Python installation...")
            python_info = self._check_python()
            
            if not python_info['installed'] or not python_info['compatible']:
                self._log(f"Python {self.required_python_major}.{self.required_python_minor}+ required")
                
                if messagebox.askyesno(
                    "Python Installation Required",
                    f"Python {self.python_download_version} needs to be installed.\n\n"
                    "This requires Administrator privileges.\n"
                    "Continue with installation?"
                ):
                    self._log("Installing Python...")
                    if not self._install_python():
                        raise Exception("Python installation failed")
                else:
                    self._log("Installation cancelled by user")
                    self._finish_installation(False)
                    return
            else:
                self._log(f"✓ Python {python_info['version']} detected")
            
            # Install packages
            self._log("")
            self._log("Installing required packages...")
            self._log("This may take 5-10 minutes...")
            
            if not self._install_packages():
                raise Exception("Package installation failed")
            
            self._log("")
            self._log("✓ Installation completed successfully!")
            self._finish_installation(True)
            
        except Exception as e:
            self._log(f"✗ Error: {str(e)}", "red")
            messagebox.showerror("Installation Failed", str(e))
            self._finish_installation(False)
    
    def _check_python(self):
        """Check if Python is installed and compatible"""
        try:
            result = subprocess.run(
                ['python', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                version_str = result.stdout.strip()
                # Parse version
                if "Python " in version_str:
                    version_str = version_str.replace("Python ", "")
                    parts = version_str.split('.')
                    major = int(parts[0])
                    minor = int(parts[1])
                    
                    compatible = (major > self.required_python_major) or \
                                (major == self.required_python_major and minor >= self.required_python_minor)
                    
                    return {
                        'installed': True,
                        'version': version_str,
                        'compatible': compatible
                    }
        except:
            pass
        
        return {'installed': False, 'compatible': False}
    
    def _install_python(self):
        """Download and install Python"""
        try:
            # Download installer
            self._log(f"Downloading Python {self.python_download_version}...")
            installer_path = Path(os.environ['TEMP']) / "python-installer.exe"
            
            urllib.request.urlretrieve(self.python_installer_url, installer_path)
            self._log("✓ Download complete")
            
            # Install Python
            self._log("Installing Python (this may take a few minutes)...")
            self._log("Please wait...")
            
            result = subprocess.run(
                [
                    str(installer_path),
                    '/quiet',
                    'InstallAllUsers=1',
                    'PrependPath=1',
                    'Include_pip=1',
                    'Include_test=0',
                    'Include_doc=0'
                ],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            # Clean up
            installer_path.unlink(missing_ok=True)
            
            if result.returncode == 0:
                self._log("✓ Python installed successfully")
                return True
            else:
                self._log(f"✗ Installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self._log(f"✗ Error installing Python: {str(e)}")
            return False
    
    def _install_packages(self):
        """Install required Python packages"""
        try:
            # Upgrade pip
            self._log("Upgrading pip...")
            result = subprocess.run(
                ['python', '-m', 'pip', 'install', '--upgrade', 'pip'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                self._log("✓ Pip upgraded")
            
            # Install from requirements.txt
            self._log("Installing packages from requirements.txt...")
            
            process = subprocess.Popen(
                ['python', '-m', 'pip', 'install', '-r', 'requirements.txt'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output if console is visible
            for line in process.stdout:
                if self.show_console.get():
                    self._log(line.rstrip())
                elif "Installing" in line or "Successfully" in line:
                    # Show key progress messages even without console
                    self._log(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                self._log("✓ All packages installed successfully")
                return True
            else:
                self._log("✗ Package installation failed")
                return False
                
        except Exception as e:
            self._log(f"✗ Error installing packages: {str(e)}")
            return False
    
    def _finish_installation(self, success):
        """Clean up after installation"""
        self.progress.stop()
        self.installation_complete = success
        
        if success:
            self.status_label.config(text="Installation Complete!", fg="green")
            self.install_button.config(text="Installation Complete", state=tk.DISABLED)
            self.close_button.config(state=tk.NORMAL)
            
            messagebox.showinfo(
                "Success",
                "Installation completed successfully!\n\n"
                "You can now run RUN_DETECTOR.bat to start the application."
            )
        else:
            self.status_label.config(text="Installation Failed", fg="red")
            self.install_button.config(text="Retry", state=tk.NORMAL)
            self.close_button.config(state=tk.NORMAL)
            
            if not self.show_console.get():
                if messagebox.askyesno(
                    "Show Details",
                    "Installation failed. Would you like to see detailed output?"
                ):
                    self.show_console.set(True)
                    self._toggle_console()

if __name__ == "__main__":
    root = tk.Tk()
    app = InstallerGUI(root)
    root.mainloop()
