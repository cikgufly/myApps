import PySimpleGUI as sg
import serial
import serial.tools.list_ports
import threading
import queue
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from collections import deque

class SerialPlotter:
    def __init__(self):
        self.serial_connection = None
        self.is_connected = False
        self.is_receiving = False
        self.data_queue = queue.Queue()
        self.headings = [] 
        self.received_data = []
        self.plot_data = {'x': deque(maxlen=1000), 'y': deque(maxlen=1000)}
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlabel('x-axis')
        self.ax.set_ylabel('y-axis')
        self.ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.last_plot_time = 0  # For controlling live plot update frequency
        
    def get_available_ports(self):
        """Get list of available serial ports"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports] if ports else ['COM1', 'COM2', 'COM3']
    
    def connect_serial(self, port, baudrate):
        """Connect to serial port"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            
            self.serial_connection = serial.Serial(port, baudrate, timeout=2)
            time.sleep(2)  # Wait for Arduino to initialize
            self.is_connected = True
            return True, "Connected successfully"
        except Exception as e:
            self.is_connected = False
            return False, f"Connection failed: {str(e)}"
    
    def request_headings(self):
        """Request column headings from Arduino"""
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                return False, [], "Not connected"
            
            # Send request for headings
            self.serial_connection.write(b"GET_HEADINGS\n")
            time.sleep(1.0)  # Wait for response
            
            # Read response
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode('utf-8').strip()
                if response:
                    # Parse headings (expected format: "A, B, C")
                    self.headings = [heading.strip() for heading in response.split(',')]
                    return True, self.headings, f"Received {len(self.headings)} self.headings"
            
            return False, [], "No response from Arduino"
            
        except Exception as e:
            return False, [], f"Error requesting headings: {str(e)}"
    
    def disconnect_serial(self):
        """Disconnect from serial port"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.is_connected = False
        self.is_receiving = False
    
    def data_receiver_thread(self, num_columns):
        """Thread function to receive data from serial port"""
        while self.is_receiving and self.is_connected:
            try:
                if self.serial_connection and self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    if line:
                        # Parse data assuming comma-separated values
                        data_parts = line.split(',')
                        if len(data_parts) >= num_columns:
                            # Convert to float, use 0 if conversion fails
                            parsed_data = []
                            for i in range(num_columns):
                                try:
                                    parsed_data.append(float(data_parts[i]))
                                except (ValueError, IndexError):
                                    parsed_data.append(0.0)
                            
                            self.data_queue.put(parsed_data)
            except Exception as e:
                print(f"Error receiving data: {e}")
                time.sleep(0.01)
            time.sleep(0.01)
    
    def start_receiving(self, num_columns):
        """Start receiving data"""
        if not self.is_connected:
            return False, "Not connected to serial port"
        
        self.is_receiving = True
        self.receiver_thread = threading.Thread(target=self.data_receiver_thread, args=(num_columns,))
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        return True, "Started receiving data"
    
    def stop_receiving(self):
        """Stop receiving data"""
        self.is_receiving = False
    
    def update_plot(self, x_col, y_col):
        """Update the plot with current data"""
        self.ax.clear()
        
        if len(self.received_data) > 0:
            df = pd.DataFrame(self.received_data)
            if x_col < len(df.columns) and y_col < len(df.columns):
                x_data = df.iloc[:, x_col].values
                y_data = df.iloc[:, y_col].values
                
                self.ax.scatter(x_data, y_data, alpha=0.7, s=30)
                self.ax.set_xlabel(f'Column {x_col + 1} (x-axis)')
                self.ax.set_ylabel(f'Column {y_col + 1} (y-axis)')
        
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw()
    
    def clear_data(self):
        """Clear all collected data"""
        self.received_data.clear()
        self.plot_data['x'].clear()
        self.plot_data['y'].clear()
        
        # Clear the plot
        self.ax.clear()
        self.ax.set_xlabel('x-axis')
        self.ax.set_ylabel('y-axis')
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw()
        
        return True, "Data cleared"
        """Save collected data to CSV file"""
        if not self.received_data:
            return False, "No data to save"
        
        try:
            df = pd.DataFrame(self.received_data)
            if headings and len(headings) == len(df.columns):
                df.columns = headings
            else:
                df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]
            df.to_csv(filename, index=False)
            return True, f"Data saved to {filename}"
        except Exception as e:
            return False, f"Error saving data: {str(e)}"

    def save_data_to_csv(self, filename, headings=None):
        """Save collected data to CSV file"""
        if not self.received_data:
            return False, "No data to save"
        
        try:
            df = pd.DataFrame(self.received_data)
            if headings and len(headings) == len(df.columns):
                df.columns = headings
            else:
                df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]
            df.to_csv(filename, index=False)
            return True, f"Data saved to {filename}"
        except Exception as e:
            return False, f"Error saving data: {str(e)}"

def draw_figure(canvas, figure):
    """Helper function to draw matplotlib figure on canvas"""
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def create_layout():
    """Create the GUI layout"""
    # Get available ports
    plotter = SerialPlotter()
    ports = plotter.get_available_ports()
    
    # Control panel
    control_frame = [
        [sg.Text('Port:', size=(12, 1)), sg.Combo(ports, default_value='COM3', key='-PORT-', size=(10, 1))],
        [sg.Text('Baudrate:', size=(12, 1)), sg.Input('9600', key='-BAUDRATE-', size=(12, 1))],
        [sg.Text('Number of Column:', size=(12, 1)), sg.Input('3', key='-NUM_COLS-', size=(12, 1), disabled=True)],
        [sg.Text('x-axis:', size=(12, 1)), sg.Combo(['Column 1', 'Column 2', 'Column 3'], default_value='Column 1', key='-X_AXIS-', size=(15, 1))],
        [sg.Text('y-axis:', size=(12, 1)), sg.Combo(['Column 1', 'Column 2', 'Column 3'], default_value='Column 2', key='-Y_AXIS-', size=(15, 1))],
        [sg.Checkbox('Live Plot', key='-LIVE_PLOT-', default=False)],
        [sg.Button('Connect', key='-CONNECT-', size=(10, 1))],
        [sg.Button('Receive', key='-RECEIVE-', size=(10, 1))],
        [sg.Button('Plot', key='-PLOT-', size=(10, 1))],
        [sg.Button('Clear Data', key='-CLEAR-', size=(10, 1))],
        [sg.Button('Save data', key='-SAVE-', size=(10, 1))],
        [sg.Text('Status:', size=(12, 1))],
        [sg.Text('Disconnected', key='-STATUS-', size=(30, 2), text_color='red')],
        [sg.Text('Value', key='-DATAVALUE-', size=(10, 1), text_color='red',background_color='white',justification='center',font=('Arial', 25))],

    ]
    
    # Plot area
    plot_frame = [
        [sg.Canvas(key='-CANVAS-', size=(600, 400), background_color='lightgreen')]
    ]
    
    # Main layout
    layout = [
        [sg.Column(control_frame, vertical_alignment='top'), 
         sg.VSeparator(),
         sg.Column(plot_frame, element_justification='center')]
    ]
    
    return layout, plotter

def main():
    sg.theme('DefaultNoMoreNagging')
    
    layout, plotter = create_layout()
    window = sg.Window('Serial Data Plotter', layout, finalize=True, resizable=True)
    
    # Draw initial plot
    fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, plotter.fig)
    
    receiving = False
    headings = []
    live_plot_enabled = False
    
    while True:
        event, values = window.read(timeout=100)
        
        if event == sg.WIN_CLOSED:
            break
        
        elif event == '-CONNECT-':
            port = values['-PORT-']
            try:
                baudrate = int(values['-BAUDRATE-'])
            except ValueError:
                baudrate = 9600
            
            if not plotter.is_connected:
                # Connect to serial port
                success, message = plotter.connect_serial(port, baudrate)
                if success:
                    window['-STATUS-'].update('Connected, requesting headings...', text_color='blue')
                    window.refresh()
                    
                    # Request headings from Arduino
                    success_headings, headings, heading_message = plotter.request_headings()
                    
                    if success_headings and headings:
                        # Update number of columns
                        window['-NUM_COLS-'].update(str(len(headings)))
                        
                        # Update dropdown menus with headings
                        window['-X_AXIS-'].update(values=headings, value=headings[0] if headings else 'Column 1')
                        window['-Y_AXIS-'].update(values=headings, value=headings[1] if len(headings) > 1 else headings[0])
                        
                        window['-STATUS-'].update(f'Connected to {port}. {heading_message}', text_color='green')
                        window['-CONNECT-'].update('Disconnect')
                    else:
                        # Fallback to default column names if no headings received
                        default_headings = [f'Column {i+1}' for i in range(3)]
                        window['-X_AXIS-'].update(values=default_headings, value='Column 1')
                        window['-Y_AXIS-'].update(values=default_headings, value='Column 2')
                        window['-STATUS-'].update(f'Connected to {port}. {heading_message}', text_color='orange')
                        window['-CONNECT-'].update('Disconnect')
                        headings = default_headings
                else:
                    window['-STATUS-'].update(message, text_color='red')
            else:
                plotter.disconnect_serial()
                window['-STATUS-'].update('Disconnected', text_color='red')
                window['-CONNECT-'].update('Connect')
                window['-RECEIVE-'].update('Receive')
                receiving = False
                
                # Reset to default values
                default_headings = ['Column 1', 'Column 2', 'Column 3']
                window['-NUM_COLS-'].update('3')
                window['-X_AXIS-'].update(values=default_headings, value='Column 1')
                window['-Y_AXIS-'].update(values=default_headings, value='Column 2')
                headings = []
        
        elif event == '-RECEIVE-':
            if not receiving and plotter.is_connected:
                try:
                    num_cols = int(values['-NUM_COLS-'])
                except ValueError:
                    num_cols = len(headings) if headings else 3
                
                success, message = plotter.start_receiving(num_cols)
                if success:
                    receiving = True
                    window['-RECEIVE-'].update('Stop')
                    window['-STATUS-'].update('Receiving data...', text_color='blue')
            else:
                plotter.stop_receiving()
                receiving = False
                window['-RECEIVE-'].update('Receive')
                if plotter.is_connected:
                    window['-STATUS-'].update(f'Connected to {values["-PORT-"]}', text_color='green')
        
        elif event == '-PLOT-':
            try:
                x_selection = values['-X_AXIS-']
                y_selection = values['-Y_AXIS-']
                
                if headings:
                    # Find index of selected headings
                    x_col = headings.index(x_selection) if x_selection in headings else 0
                    y_col = headings.index(y_selection) if y_selection in headings else 1
                else:
                    # Fallback to numeric parsing
                    x_col = int(x_selection.split()[-1]) - 1 if 'Column' in x_selection else 0
                    y_col = int(y_selection.split()[-1]) - 1 if 'Column' in y_selection else 1
                
                plotter.update_plot(x_col, y_col)
                fig_agg.draw()
            except Exception as e:
                window['-STATUS-'].update(f'Plot error: {str(e)}', text_color='red')
        
        elif event == '-CLEAR-':
            success, message = plotter.clear_data()
            fig_agg.draw()
            window['-STATUS-'].update(message, text_color='green' if success else 'red')
        
        elif event == '-SAVE-':
            if plotter.received_data:
                filename = sg.popup_get_file('Save data as...', save_as=True, 
                                           file_types=(('CSV Files', '*.csv'),))
                if filename:
                    if not filename.endswith('.csv'):
                        filename += '.csv'
                    success, message = plotter.save_data_to_csv(filename, headings if headings else None)
                    window['-STATUS-'].update(message, 
                                            text_color='green' if success else 'red')
            else:
                window['-STATUS-'].update('No data to save', text_color='orange')
        
        # Check live plot status
        live_plot_enabled = values['-LIVE_PLOT-']
        
        # Process received data
        if receiving:
            data_count = 0
            while not plotter.data_queue.empty() and data_count < 10:  # Process max 10 items per cycle
                try:
                    data = plotter.data_queue.get_nowait()
                    plotter.received_data.append(data)
                    data_count += 1
                except queue.Empty:
                    break
            
            # Update live plot if enabled and new data received
            if live_plot_enabled and data_count>0:
                try:
                    x_selection = values['-X_AXIS-']
                    y_selection = values['-Y_AXIS-']
                    
                    if headings:
                        x_col = headings.index(x_selection) if x_selection in headings else 0
                        y_col = headings.index(y_selection) if y_selection in headings else 1
                    else:
                        x_col = int(x_selection.split()[-1]) - 1 if 'Column' in x_selection else 0
                        y_col = int(y_selection.split()[-1]) - 1 if 'Column' in y_selection else 1
                    
                    plotter.update_plot(x_col, y_col)
                except Exception as e:
                    pass  # Silently handle live plot errors to avoid spam
            
            # Update status with data count
            if plotter.received_data:
                status_text = f'Receiving... ({len(plotter.received_data)} samples)'
                if live_plot_enabled:
                    status_text += ' [Live Plot ON]'
                window['-STATUS-'].update(status_text, text_color='blue')
    
    # Cleanup
    plotter.disconnect_serial()
    window.close()

if __name__ == '__main__':
    main()
