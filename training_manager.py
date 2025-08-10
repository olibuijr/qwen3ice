#!/usr/bin/env python3
"""
Qwen Icelandic Training Manager - Comprehensive CLI GUI
A full-featured terminal UI for managing and monitoring LLM training
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import queue
import subprocess
import psutil

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich.chart import BarChart
from rich.tree import Tree
from rich import box
from rich.columns import Columns
from rich.align import Align

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Button, Header, Footer, Static, Label, ListView, ListItem, ProgressBar, Log, TabbedContent, TabPane, DataTable, Input, TextArea, Sparkline, RichLog
from textual.reactive import reactive
from textual.binding import Binding
from textual.timer import Timer
from textual.screen import Screen
from textual import events

import click
import plotext as plt

# Import training components
try:
    import torch
    import nvidia_ml_py as nvml
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


class TrainingMetrics:
    """Collects and manages training metrics"""
    
    def __init__(self):
        self.metrics = {
            'loss': [],
            'learning_rate': [],
            'throughput': [],
            'gpu_memory': [],
            'gpu_utilization': [],
            'cpu_usage': [],
            'ram_usage': [],
            'temperature': [],
            'time': [],
            'epoch': 0,
            'step': 0,
            'total_steps': 0,
        }
        self.start_time = time.time()
    
    def update(self, **kwargs):
        """Update metrics with new values"""
        current_time = time.time() - self.start_time
        self.metrics['time'].append(current_time)
        
        for key, value in kwargs.items():
            if key in self.metrics:
                if isinstance(self.metrics[key], list):
                    self.metrics[key].append(value)
                else:
                    self.metrics[key] = value
    
    def get_latest(self, metric: str, default=0):
        """Get latest value for a metric"""
        if metric in self.metrics:
            if isinstance(self.metrics[metric], list) and self.metrics[metric]:
                return self.metrics[metric][-1]
            elif not isinstance(self.metrics[metric], list):
                return self.metrics[metric]
        return default
    
    def get_history(self, metric: str, last_n: int = 50):
        """Get history of a metric"""
        if metric in self.metrics and isinstance(self.metrics[metric], list):
            return self.metrics[metric][-last_n:]
        return []


class SystemMonitor:
    """Monitors system resources"""
    
    def __init__(self):
        self.cpu_percent = 0
        self.ram_percent = 0
        self.gpu_memory = 0
        self.gpu_util = 0
        self.gpu_temp = 0
        self.disk_usage = 0
        
        if CUDA_AVAILABLE:
            nvml.nvmlInit()
            self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
    
    def update(self):
        """Update system metrics"""
        # CPU and RAM
        self.cpu_percent = psutil.cpu_percent(interval=0.1)
        self.ram_percent = psutil.virtual_memory().percent
        
        # Disk
        disk = psutil.disk_usage('/')
        self.disk_usage = disk.percent
        
        # GPU (if available)
        if CUDA_AVAILABLE:
            try:
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.gpu_memory = (mem_info.used / mem_info.total) * 100
                
                util = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                self.gpu_util = util.gpu
                
                temp = nvml.nvmlDeviceGetTemperature(self.gpu_handle, nvml.NVML_TEMPERATURE_GPU)
                self.gpu_temp = temp
            except:
                pass
    
    def get_stats(self) -> Dict:
        """Get current system stats"""
        self.update()
        return {
            'cpu': self.cpu_percent,
            'ram': self.ram_percent,
            'gpu_memory': self.gpu_memory,
            'gpu_util': self.gpu_util,
            'gpu_temp': self.gpu_temp,
            'disk': self.disk_usage,
        }


class TrainingController:
    """Controls the training process"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.process = None
        self.is_running = False
        self.is_paused = False
        self.output_queue = queue.Queue()
        self.metrics = TrainingMetrics()
        self.monitor = SystemMonitor()
        self.log_file = None
        self.current_checkpoint = None
        
    def load_config(self) -> Dict:
        """Load training configuration"""
        import yaml
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config: Dict):
        """Save training configuration"""
        import yaml
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def start_training(self, resume_from: Optional[str] = None):
        """Start the training process"""
        if self.is_running:
            return False
        
        cmd = ["python", "train_qwen_icelandic.py"]
        if resume_from:
            cmd.extend(["--resume", resume_from])
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.is_running = True
        self.is_paused = False
        
        # Start output reader thread
        threading.Thread(target=self._read_output, daemon=True).start()
        
        # Open log file
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(log_dir / f"training_{timestamp}.log", 'w')
        
        return True
    
    def _read_output(self):
        """Read training process output"""
        while self.is_running and self.process:
            line = self.process.stdout.readline()
            if line:
                self.output_queue.put(line.strip())
                if self.log_file:
                    self.log_file.write(line)
                    self.log_file.flush()
                
                # Parse metrics from output
                self._parse_metrics(line)
            
            if self.process.poll() is not None:
                self.is_running = False
                break
    
    def _parse_metrics(self, line: str):
        """Parse metrics from training output"""
        # Parse loss
        if "loss:" in line.lower():
            try:
                loss_str = line.split("loss:")[1].split()[0]
                loss = float(loss_str.replace(',', ''))
                self.metrics.update(loss=loss)
            except:
                pass
        
        # Parse learning rate
        if "lr:" in line.lower() or "learning_rate:" in line.lower():
            try:
                lr_str = line.split("lr:")[1].split()[0] if "lr:" in line else line.split("learning_rate:")[1].split()[0]
                lr = float(lr_str.replace(',', ''))
                self.metrics.update(learning_rate=lr)
            except:
                pass
        
        # Parse epoch
        if "epoch" in line.lower():
            try:
                if "epoch:" in line.lower():
                    epoch_str = line.split("epoch:")[1].split()[0]
                    self.metrics.metrics['epoch'] = int(epoch_str.replace(',', ''))
            except:
                pass
        
        # Parse step
        if "step" in line.lower():
            try:
                if "step:" in line.lower():
                    step_str = line.split("step:")[1].split()[0]
                    self.metrics.metrics['step'] = int(step_str.replace(',', ''))
            except:
                pass
    
    def stop_training(self):
        """Stop the training process"""
        if self.process:
            self.process.terminate()
            self.process = None
        self.is_running = False
        self.is_paused = False
        if self.log_file:
            self.log_file.close()
            self.log_file = None
    
    def pause_training(self):
        """Pause training (save checkpoint)"""
        if self.is_running and not self.is_paused:
            # Send signal to save checkpoint
            if self.process:
                self.process.send_signal(signal.SIGUSR1)  # Custom signal for checkpoint
            self.is_paused = True
    
    def resume_training(self):
        """Resume training from checkpoint"""
        if self.is_paused:
            # Find latest checkpoint
            checkpoint_dir = Path("checkpoints")
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
                if checkpoints:
                    self.current_checkpoint = str(checkpoints[-1])
                    self.stop_training()
                    self.start_training(resume_from=self.current_checkpoint)
    
    def get_logs(self, last_n: int = 100) -> List[str]:
        """Get recent log lines"""
        logs = []
        while not self.output_queue.empty() and len(logs) < last_n:
            logs.append(self.output_queue.get())
        return logs


class TrainingManagerApp(App):
    """Main Training Manager Application"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #header {
        height: 3;
        background: $primary;
        color: $text;
        content-align: center middle;
    }
    
    #sidebar {
        width: 30;
        background: $panel;
        border-right: solid $primary;
    }
    
    #main-content {
        background: $surface;
    }
    
    .metric-card {
        height: 7;
        margin: 1;
        padding: 1;
        background: $panel;
        border: solid $primary;
    }
    
    Button {
        margin: 1;
    }
    
    .button-primary {
        background: $success;
    }
    
    .button-danger {
        background: $error;
    }
    
    ProgressBar {
        height: 3;
        margin: 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "start_training", "Start Training"),
        Binding("x", "stop_training", "Stop Training"),
        Binding("p", "pause_training", "Pause"),
        Binding("r", "resume_training", "Resume"),
        Binding("l", "show_logs", "Logs"),
        Binding("m", "show_metrics", "Metrics"),
        Binding("c", "show_config", "Config"),
        Binding("d", "show_dataset", "Dataset"),
        Binding("h", "show_help", "Help"),
    ]
    
    def __init__(self):
        super().__init__()
        self.controller = TrainingController()
        self.update_timer = None
        self.current_tab = "overview"
    
    def compose(self) -> ComposeResult:
        """Create the UI layout"""
        yield Header(show_clock=True)
        
        with Container():
            with Horizontal():
                # Sidebar
                with Vertical(id="sidebar"):
                    yield Label("🚀 Training Control", classes="title")
                    yield Button("▶️ Start Training", id="btn-start", variant="primary")
                    yield Button("⏸️ Pause", id="btn-pause", variant="default")
                    yield Button("⏹️ Stop", id="btn-stop", variant="error")
                    yield Button("📊 Metrics", id="btn-metrics")
                    yield Button("📜 Logs", id="btn-logs")
                    yield Button("⚙️ Config", id="btn-config")
                    yield Button("💾 Dataset", id="btn-dataset")
                    yield Button("🔍 Monitor", id="btn-monitor")
                    yield Button("❓ Help", id="btn-help")
                
                # Main content area with tabs
                with TabbedContent(initial="overview", id="main-content"):
                    with TabPane("Overview", id="overview"):
                        yield self._create_overview_tab()
                    
                    with TabPane("Metrics", id="metrics"):
                        yield self._create_metrics_tab()
                    
                    with TabPane("Logs", id="logs"):
                        yield self._create_logs_tab()
                    
                    with TabPane("Config", id="config"):
                        yield self._create_config_tab()
                    
                    with TabPane("Dataset", id="dataset"):
                        yield self._create_dataset_tab()
                    
                    with TabPane("Monitor", id="monitor"):
                        yield self._create_monitor_tab()
                    
                    with TabPane("Help", id="help"):
                        yield self._create_help_tab()
        
        yield Footer()
    
    def _create_overview_tab(self) -> Container:
        """Create overview tab content"""
        return Container(
            Label("📊 Training Overview", classes="title"),
            Static(id="training-status"),
            ProgressBar(id="training-progress", total=100),
            Static(id="current-metrics"),
            Static(id="system-stats"),
            Sparkline(id="loss-sparkline", data=[]),
            id="overview-content"
        )
    
    def _create_metrics_tab(self) -> Container:
        """Create metrics tab content"""
        return ScrollableContainer(
            Label("📈 Training Metrics", classes="title"),
            DataTable(id="metrics-table"),
            Static(id="loss-chart"),
            Static(id="lr-chart"),
            Static(id="throughput-chart"),
            id="metrics-content"
        )
    
    def _create_logs_tab(self) -> Container:
        """Create logs tab content"""
        return Container(
            Label("📜 Training Logs", classes="title"),
            RichLog(id="log-viewer", wrap=True, highlight=True, markup=True),
            Input(placeholder="Filter logs...", id="log-filter"),
            id="logs-content"
        )
    
    def _create_config_tab(self) -> Container:
        """Create configuration tab content"""
        return ScrollableContainer(
            Label("⚙️ Configuration", classes="title"),
            TextArea(id="config-editor", language="yaml"),
            Button("💾 Save Config", id="save-config"),
            Button("🔄 Reload Config", id="reload-config"),
            id="config-content"
        )
    
    def _create_dataset_tab(self) -> Container:
        """Create dataset tab content"""
        return Container(
            Label("💾 Dataset Information", classes="title"),
            Static(id="dataset-stats"),
            DataTable(id="dataset-samples"),
            Button("🔄 Refresh Dataset", id="refresh-dataset"),
            id="dataset-content"
        )
    
    def _create_monitor_tab(self) -> Container:
        """Create system monitor tab content"""
        return Container(
            Label("🖥️ System Monitor", classes="title"),
            Static(id="gpu-stats"),
            Static(id="cpu-stats"),
            Static(id="memory-stats"),
            Sparkline(id="gpu-usage-sparkline", data=[]),
            Sparkline(id="cpu-usage-sparkline", data=[]),
            id="monitor-content"
        )
    
    def _create_help_tab(self) -> Container:
        """Create help tab content"""
        help_text = """
# 🚀 Qwen Icelandic Training Manager

## Keyboard Shortcuts:
- `s` - Start Training
- `x` - Stop Training  
- `p` - Pause Training
- `r` - Resume Training
- `l` - Show Logs
- `m` - Show Metrics
- `c` - Show Config
- `d` - Show Dataset
- `h` - Show Help
- `q` - Quit

## Features:
- Real-time training metrics monitoring
- GPU/CPU/Memory usage tracking
- Live loss graphs and visualizations
- Log filtering and searching
- Configuration editing
- Dataset preview and statistics
- Checkpoint management
- Training pause/resume support

## Training States:
- 🟢 Running - Training is active
- 🟡 Paused - Training paused (checkpoint saved)
- 🔴 Stopped - Training stopped
- ⚪ Ready - Ready to start training

## Tips:
- Monitor GPU memory to avoid OOM errors
- Use pause/resume for long training sessions
- Check logs for detailed error messages
- Adjust config for better performance
        """
        
        return ScrollableContainer(
            Static(help_text, markup=True),
            id="help-content"
        )
    
    def on_mount(self) -> None:
        """Called when app starts"""
        self.update_timer = self.set_interval(1.0, self.update_display)
        self.load_initial_data()
    
    def load_initial_data(self):
        """Load initial configuration and dataset info"""
        # Load config
        try:
            config = self.controller.load_config()
            config_editor = self.query_one("#config-editor", TextArea)
            import yaml
            config_editor.text = yaml.dump(config, default_flow_style=False)
        except:
            pass
        
        # Load dataset stats
        self.update_dataset_stats()
    
    def update_display(self):
        """Update the display with latest metrics"""
        if not self.controller.is_running:
            status = "🔴 Stopped"
        elif self.controller.is_paused:
            status = "🟡 Paused"
        else:
            status = "🟢 Running"
        
        # Update status
        status_widget = self.query_one("#training-status", Static)
        status_widget.update(f"Status: {status}")
        
        # Update metrics
        metrics = self.controller.metrics
        sys_stats = self.controller.monitor.get_stats()
        
        # Update current metrics display
        metrics_text = f"""
Current Metrics:
├─ Loss: {metrics.get_latest('loss'):.4f}
├─ Learning Rate: {metrics.get_latest('learning_rate'):.6f}
├─ Epoch: {metrics.metrics['epoch']}/{3}
├─ Step: {metrics.metrics['step']}
└─ Throughput: {metrics.get_latest('throughput'):.2f} samples/sec
        """
        
        current_metrics = self.query_one("#current-metrics", Static)
        current_metrics.update(metrics_text)
        
        # Update system stats
        system_text = f"""
System Resources:
├─ GPU Memory: {sys_stats['gpu_memory']:.1f}%
├─ GPU Utilization: {sys_stats['gpu_util']:.1f}%
├─ GPU Temperature: {sys_stats['gpu_temp']}°C
├─ CPU Usage: {sys_stats['cpu']:.1f}%
├─ RAM Usage: {sys_stats['ram']:.1f}%
└─ Disk Usage: {sys_stats['disk']:.1f}%
        """
        
        system_stats = self.query_one("#system-stats", Static)
        system_stats.update(system_text)
        
        # Update progress bar
        if metrics.metrics['total_steps'] > 0:
            progress = (metrics.metrics['step'] / metrics.metrics['total_steps']) * 100
            progress_bar = self.query_one("#training-progress", ProgressBar)
            progress_bar.update(progress=progress)
        
        # Update sparklines
        loss_history = metrics.get_history('loss', 50)
        if loss_history:
            loss_sparkline = self.query_one("#loss-sparkline", Sparkline)
            loss_sparkline.data = loss_history
        
        # Update logs
        logs = self.controller.get_logs(50)
        if logs:
            log_viewer = self.query_one("#log-viewer", RichLog)
            for log in logs:
                log_viewer.write(log)
    
    def update_dataset_stats(self):
        """Update dataset statistics"""
        try:
            stats_text = """
Dataset Statistics:
├─ Training Examples: 10,121
├─ Validation Examples: 533
├─ Total Examples: 10,654
├─ Average Length: ~500 tokens
└─ Languages: Icelandic (is)

Data Sources:
├─ Wikipedia: 3,844 examples
├─ IC3: 4,910 examples
└─ Wiki QA: 1,900 examples
            """
            
            dataset_stats = self.query_one("#dataset-stats", Static)
            dataset_stats.update(stats_text)
            
            # Load sample data
            samples_table = self.query_one("#dataset-samples", DataTable)
            samples_table.add_columns("ID", "Type", "Length", "Preview")
            
            # Add sample rows
            import json
            with open("data/icelandic/train.jsonl", 'r') as f:
                for i, line in enumerate(f):
                    if i >= 5:  # Show first 5 samples
                        break
                    sample = json.loads(line)
                    preview = sample['messages'][1]['content'][:50] + "..."
                    samples_table.add_row(str(i), "train", str(len(str(sample))), preview)
        except:
            pass
    
    def action_start_training(self):
        """Start training action"""
        self.controller.start_training()
        self.notify("Training started!", severity="success")
    
    def action_stop_training(self):
        """Stop training action"""
        self.controller.stop_training()
        self.notify("Training stopped!", severity="warning")
    
    def action_pause_training(self):
        """Pause training action"""
        self.controller.pause_training()
        self.notify("Training paused!", severity="information")
    
    def action_resume_training(self):
        """Resume training action"""
        self.controller.resume_training()
        self.notify("Training resumed!", severity="success")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        button_id = event.button.id
        
        if button_id == "btn-start":
            self.action_start_training()
        elif button_id == "btn-stop":
            self.action_stop_training()
        elif button_id == "btn-pause":
            self.action_pause_training()
        elif button_id == "save-config":
            self.save_configuration()
        elif button_id == "reload-config":
            self.load_initial_data()
        elif button_id == "refresh-dataset":
            self.update_dataset_stats()
        
        # Tab navigation
        tab_map = {
            "btn-metrics": "metrics",
            "btn-logs": "logs",
            "btn-config": "config",
            "btn-dataset": "dataset",
            "btn-monitor": "monitor",
            "btn-help": "help",
        }
        
        if button_id in tab_map:
            tabbed_content = self.query_one("#main-content", TabbedContent)
            tabbed_content.active = tab_map[button_id]
    
    def save_configuration(self):
        """Save the configuration"""
        try:
            config_editor = self.query_one("#config-editor", TextArea)
            import yaml
            config = yaml.safe_load(config_editor.text)
            self.controller.save_config(config)
            self.notify("Configuration saved!", severity="success")
        except Exception as e:
            self.notify(f"Error saving config: {e}", severity="error")


class CLIManager:
    """CLI interface for the training manager"""
    
    @click.group()
    def cli():
        """Qwen Icelandic Training Manager CLI"""
        pass
    
    @cli.command()
    @click.option('--config', default='config.yaml', help='Configuration file path')
    def gui(config):
        """Launch the GUI training manager"""
        app = TrainingManagerApp()
        app.run()
    
    @cli.command()
    @click.option('--config', default='config.yaml', help='Configuration file path')
    @click.option('--resume', default=None, help='Resume from checkpoint')
    def start(config, resume):
        """Start training from CLI"""
        controller = TrainingController(config)
        controller.start_training(resume_from=resume)
        
        print("Training started. Press Ctrl+C to stop.")
        try:
            while controller.is_running:
                time.sleep(1)
                # Print latest metrics
                metrics = controller.metrics
                sys_stats = controller.monitor.get_stats()
                
                print(f"\rLoss: {metrics.get_latest('loss'):.4f} | "
                      f"GPU: {sys_stats['gpu_memory']:.1f}% | "
                      f"Step: {metrics.metrics['step']}", end='')
        except KeyboardInterrupt:
            controller.stop_training()
            print("\nTraining stopped.")
    
    @cli.command()
    def monitor():
        """Monitor system resources"""
        monitor = SystemMonitor()
        
        print("System Monitor (Press Ctrl+C to stop)")
        print("-" * 50)
        
        try:
            while True:
                stats = monitor.get_stats()
                print(f"\rCPU: {stats['cpu']:.1f}% | "
                      f"RAM: {stats['ram']:.1f}% | "
                      f"GPU: {stats['gpu_memory']:.1f}% | "
                      f"GPU Temp: {stats['gpu_temp']}°C", end='')
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    
    @cli.command()
    @click.argument('log_file')
    def analyze(log_file):
        """Analyze training logs"""
        print(f"Analyzing {log_file}...")
        
        losses = []
        steps = []
        
        with open(log_file, 'r') as f:
            for line in f:
                if 'loss:' in line.lower():
                    try:
                        loss = float(line.split('loss:')[1].split()[0].replace(',', ''))
                        losses.append(loss)
                        steps.append(len(losses))
                    except:
                        pass
        
        if losses:
            print(f"Total steps: {len(losses)}")
            print(f"Initial loss: {losses[0]:.4f}")
            print(f"Final loss: {losses[-1]:.4f}")
            print(f"Min loss: {min(losses):.4f}")
            print(f"Max loss: {max(losses):.4f}")
            
            # Plot with plotext
            plt.plot(steps, losses)
            plt.title("Training Loss")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.show()
        else:
            print("No loss data found in log file.")


def main():
    """Main entry point"""
    # Check if running in terminal that supports TUI
    if os.environ.get('TERM'):
        cli = CLIManager()
        cli.cli()
    else:
        print("Please run in a terminal that supports TUI applications.")
        sys.exit(1)


if __name__ == "__main__":
    main()