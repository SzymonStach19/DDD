"""
GUI components for drowsiness detection
"""
import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

class DrowsinessAnalyzerGUI:
    def __init__(self, analyzer):
        """Initializes graphical interface with detection mode options"""
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("Drowsiness Analyzer - Photo Analysis")
        self.root.geometry("900x700")
        
        # Variable for detection mode
        self.detection_mode = tk.StringVar(value='both')
        
        self.setup_gui()
    
    def setup_gui(self):
        """Configures graphical interface"""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame for detection mode options
        mode_frame = tk.LabelFrame(main_frame, text="Detection Mode", font=("Arial", 12))
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Radio buttons for mode selection
        modes = [
            ('Eyes only', 'eyes'),
            ('Yawning only', 'yawn'),
            ('Eyes and yawning', 'both')
        ]
        
        for text, value in modes:
            rb = tk.Radiobutton(mode_frame, text=text, variable=self.detection_mode, 
                               value=value, font=("Arial", 10))
            rb.pack(side=tk.LEFT, padx=20, pady=10)
        
        # File selection button
        select_button = tk.Button(main_frame, text="Select Photo", 
                                 command=self.select_image, 
                                 font=("Arial", 12), 
                                 bg="#4CAF50", fg="white",
                                 height=2, width=20)
        select_button.pack(pady=10)
        
        # Frame for image
        self.image_frame = tk.Frame(main_frame, bg="lightgray", height=400)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Label for image
        self.image_label = tk.Label(self.image_frame, text="Select photo for analysis", 
                                   font=("Arial", 14), bg="lightgray")
        self.image_label.pack(expand=True)
        
        # Frame for results
        self.results_frame = tk.Frame(main_frame, height=100)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Text widget for results
        self.results_text = tk.Text(self.results_frame, font=("Arial", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for results
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
    
    def select_image(self):
        """Opens file selection dialog and analyzes image"""
        file_path = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.analyze_selected_image(file_path)
    
    def analyze_selected_image(self, image_path):
        """Analyzes selected image with chosen detection mode"""
        try:
            # Get selected detection mode
            mode = self.detection_mode.get()
            
            # Image analysis
            result_image, analysis_results = self.analyzer.analyze_image(image_path, mode)
            
            if isinstance(analysis_results, str):  # Error
                messagebox.showerror("Error", analysis_results)
                return
            
            # Display result image
            self.display_image(result_image)
            
            # Display analysis results
            self.display_results(analysis_results, image_path, mode)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
    
    def display_image(self, cv_image):
        """Displays image in GUI"""
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize image for display
        height, width = rgb_image.shape[:2]
        max_height = 400
        max_width = 600
        
        if height > max_height or width > max_width:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display in label
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo  # Keep reference
    
    def display_results(self, analysis_results, image_path, detection_mode):
        """Displays analysis results with mode information"""
        self.results_text.delete(1.0, tk.END)
        
        mode_names = {
            'eyes': 'EYES ONLY',
            'yawn': 'YAWNING ONLY', 
            'both': 'EYES AND YAWNING'
        }
        
        results_text = f"IMAGE ANALYSIS: {os.path.basename(image_path)}\n"
        results_text += f"DETECTION MODE: {mode_names[detection_mode]}\n"
        results_text += "=" * 50 + "\n\n"
        
        if not analysis_results:
            results_text += "No faces detected in image.\n"
        else:
            for result in analysis_results:
                results_text += f"FACE {result['face_id']}:\n"
                if detection_mode in ['eyes', 'both']:
                    results_text += f"  Eye state: {result['eye_state']}"
                    if result['confidence_eye'] > 0:
                        results_text += f" (confidence: {result['confidence_eye']:.2%})"
                    results_text += "\n"
                if detection_mode in ['yawn', 'both']:
                    results_text += f"  Yawn state: {result['yawn_state']}"
                    results_text += f" (confidence: {result['confidence_yawn']:.2%})\n"
                results_text += f"  State: {result['drowsiness_level']}\n"
                results_text += "-" * 30 + "\n"
        self.results_text.insert(1.0, results_text)
    
    def run(self):
        """Runs GUI"""
        self.root.mainloop()
