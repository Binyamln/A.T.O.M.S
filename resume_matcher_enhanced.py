#!/usr/bin/env python3

import os
import sys
import fitz  # PyMuPDF
import re
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, Toplevel
from tkinter.scrolledtext import ScrolledText
from sentence_transformers import SentenceTransformer, util
import threading
import time
from PIL import Image, ImageTk # Import Pillow modules
import subprocess # Added for opening files cross-platform

# Define application data directory
APP_DATA_DIR = os.path.join(os.path.expanduser("~"), ".resume_ranker")
MODEL_DIR = os.path.join(APP_DATA_DIR, "model")

# Create necessary directories
os.makedirs(APP_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Store root reference globally or pass it around
# This helps in managing window transitions
app_root = None 

# --- Global Theme Settings ---
# Define professional color palette
bg_color = "#F0F2F5"       # Light grey background
accent_color = "#0078D4"   # Professional blue accent
accent_hover_color = "#005A9E" # Darker blue for hover
text_color = "#1F1F1F"     # Dark grey text
card_bg = "#FFFFFF"       # White card background
border_color = "#D9D9D9"   # Light grey border

# Define professional fonts
font_family = "Segoe UI"
font_normal = (font_family, 10)
font_bold = (font_family, 10, "bold")
font_large_bold = (font_family, 16, "bold")
font_medium_bold = (font_family, 12, "bold")
# --- End Global Theme Settings ---

def extract_text(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def normalize_text(text):
    """Normalize the text by removing unnecessary spaces and collapsing whitespace."""
    # Remove letter-by-letter spaced out words
    text = remove_spaced_out_words(text)
    # Strip and collapse whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_spaced_out_words(text):
    """Find sequences where letters (A-Z) are separated by spaces and remove those spaces."""
    pattern = r'(?<!\S)(?:[A-Z]\s)+(?:[A-Z])(?!\S)'
    return re.sub(pattern, lambda m: m.group(0).replace(" ", ""), text)

def extract_candidate_name(text):
    """Extract the candidate name from the resume text."""
    for line in text.splitlines():
        stripped_line = line.strip()
        if stripped_line:
            return stripped_line
    return ""

def load_model(progress_callback=None):
    """Load or download the sentence-transformer model with progress updates."""
    model_path = os.path.join(MODEL_DIR, "all-MiniLM-L6-v2")
    
    # Check if model exists
    if not os.path.exists(model_path):
        if progress_callback:
            progress_callback(0.0, "Downloading model for first use. This may take a few minutes...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.save(model_path)
        if progress_callback:
            progress_callback(1.0, "Model downloaded successfully")
    else:
        # Load the model from disk
        if progress_callback:
            progress_callback(0.0, "Loading model from disk...")
            # Simulate progress for model loading
            for i in range(1, 11):
                time.sleep(0.1)  # Small delay to simulate loading
                progress_callback(i/10, f"Loading model components ({i*10}%)...")
        model = SentenceTransformer(model_path)
        if progress_callback:
            progress_callback(1.0, "Model loaded successfully")
    
    return model

def rank_resumes(resumes, job_description, model, progress_callback=None):
    """Rank resumes based on similarity to job description with progress updates."""
    # Encode the job description
    if progress_callback:
        progress_callback(0.0, "Encoding job description...")
    
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    
    if progress_callback:
        progress_callback(0.2, "Encoding resumes...")
    
    # Extract resume texts
    resume_texts = [resume['text'] for resume in resumes]
    
    # Encode all resume texts
    resume_embeddings = model.encode(resume_texts, convert_to_tensor=True, show_progress_bar=False)
    
    if progress_callback:
        progress_callback(0.7, "Computing similarity scores...")
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(job_embedding, resume_embeddings)[0]
    
    if progress_callback:
        progress_callback(0.9, "Ranking resumes...")
    
    # Zip and rank
    ranked_resumes = sorted(zip(resumes, cosine_scores.cpu().numpy()), 
                           key=lambda x: x[1], reverse=True)
    
    if progress_callback:
        progress_callback(1.0, "Ranking complete")
    
    return ranked_resumes

class LandingScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Matcher - Welcome")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)
        self.root.configure(bg=bg_color)
        
        self.main_frame = tk.Frame(root, bg=bg_color, padx=20, pady=20)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        # Title Label
        title_label = tk.Label(self.main_frame, text="Select Resume Source", 
                               font=font_large_bold, bg=bg_color, fg=text_color)
        title_label.pack(pady=(10, 30))

        # Button Style
        button_style = {'font': font_bold, 'bg': accent_color, 'fg': 'white', 
                        'activebackground': accent_hover_color, 'activeforeground': 'white',
                        'width': 25, 'pady': 10, 'bd': 0, 'relief': tk.FLAT}

        # Add Resumes Individually Button
        individual_button = tk.Button(self.main_frame, text="Add Resumes Individually", 
                                      command=self.add_individually, **button_style)
        individual_button.pack(pady=10)
        
        # Add Resumes From Folder Button
        folder_button = tk.Button(self.main_frame, text="Add Resumes From Folder", 
                                  command=self.add_from_folder, **button_style)
        folder_button.pack(pady=10)
        
        # Center the frame content
        self.main_frame.pack_propagate(False) # Prevent frame from shrinking to content
        self.main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=350, height=250)


    def add_individually(self):
        # Hide the landing screen frame
        self.main_frame.pack_forget() 
        # Destroy the landing screen widgets to clean up
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.main_frame.destroy()
        
        # Initialize and show the individual resume screen
        IndividualResumeScreen(self.root)

    def add_from_folder(self):
        # Hide the landing screen frame
        self.main_frame.pack_forget() 
        # Destroy the landing screen widgets to clean up
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.main_frame.destroy()
        
        # Initialize and show the main application GUI
        # Pass the same root window
        EnhancedResumeRankerGUI(self.root) 

class IndividualResumeScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Matcher - Add Resumes Individually")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)
        self.root.configure(bg=bg_color)
        
        # Variables
        self.candidate_name = tk.StringVar()
        self.selected_file = None
        self.resume_list = []  # Store added resumes
        self.job_role = tk.StringVar()
        
        # Pagination and results variables
        self.results_per_page = 10
        self.current_page = 1
        self.total_pages = 1
        self.all_ranked_resumes = []  # Store all results
        self.is_analyzed = False  # Track if analysis has been performed
        
        # Load Logo
        self.logo_image = None
        try:
            # Determine base path whether running as script or frozen executable
            if getattr(sys, 'frozen', False):
                # If the application is run as a bundle/executable, the base path is the sys._MEIPASS
                base_path = sys._MEIPASS
            else:
                # If run as a script, the base path is the script directory
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            # Define the logo path relative to the base path
            logo_path = os.path.join(base_path, "public/images", "ATOMS_LOGO.png")
            
            if os.path.exists(logo_path):
                img = Image.open(logo_path)
                # Resize logo to a suitable size (e.g., height 60 pixels)
                img_height = 60
                img_ratio = img.height / img_height
                img_width = int(img.width / img_ratio)
                img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                self.logo_image = ImageTk.PhotoImage(img)
            else:
                print(f"Warning: Logo file not found at expected location: {logo_path}")
        except Exception as e:
            print(f"Error loading logo: {e}")
        
        # Set theme
        self.set_theme()
        
        # Create UI
        self.create_ui()
    
    def set_theme(self):
        """Set the application color scheme and styling for a professional look"""
        # Use global theme settings
        self.bg_color = bg_color
        self.accent_color = accent_color
        self.accent_hover_color = accent_hover_color
        self.text_color = text_color
        self.secondary_text_color = "#595959" # Can remain specific if needed
        self.card_bg = card_bg
        self.border_color = border_color
        
        self.font_family = font_family
        self.font_normal = font_normal
        self.font_bold = font_bold
        self.font_small = (self.font_family, 9) # Can remain specific
        self.font_large_bold = font_large_bold
        self.font_medium_bold = font_medium_bold

        # Configure root window
        self.root.configure(bg=self.bg_color)
        
        # Define styles for ttk widgets
        self.style = ttk.Style()
        self.style.theme_use('clam') # Use a theme that allows more customization

        # General Frame Style
        self.style.configure("TFrame", background=self.bg_color)
        
        # Card Style Frame
        self.style.configure("Card.TFrame", 
                             background=self.card_bg,
                             borderwidth=1,
                             relief="solid")
        self.style.map("Card.TFrame", bordercolor=[("!focus", self.border_color)])
        
        # Frame for ScrolledText border
        self.style.configure("TextBorder.TFrame", 
                             background=self.card_bg, # Match card background
                             borderwidth=1, 
                             relief="solid")
        self.style.map("TextBorder.TFrame", bordercolor=[("!focus", self.border_color)])

        # Label Styles
        self.style.configure("TLabel", 
                             background=self.bg_color, 
                             foreground=self.text_color,
                             font=self.font_normal)
        
        self.style.configure("Card.TLabel", 
                             background=self.card_bg, 
                             foreground=self.text_color,
                             font=self.font_normal)

        self.style.configure("Header.TLabel", 
                             background=self.bg_color, 
                             foreground=self.text_color,
                             font=self.font_large_bold)
        
        self.style.configure("SubHeader.TLabel", 
                             background=self.card_bg, 
                             foreground=self.text_color,
                             font=self.font_medium_bold)
        
        self.style.configure("Secondary.TLabel", 
                             background=self.card_bg, 
                             foreground=self.secondary_text_color,
                             font=self.font_small)

        # Button Style
        self.style.configure("TButton", 
                             font=self.font_bold,
                             background=self.accent_color,
                             foreground="white",
                             borderwidth=0,
                             padding=(14, 9), # Slightly increased padding
                             relief="flat",
                             cursor="hand2") # Add cursor directly here 
        self.style.map("TButton",
                       background=[('active', self.accent_hover_color), 
                                   ('disabled', '#B0B0B0'), # Explicit disabled background
                                   ('!disabled', self.accent_color)],
                       foreground=[('active', 'white'), ('disabled', '#F0F0F0')])
        
        # Entry Style
        self.style.configure("TEntry", 
                             fieldbackground="white",
                             borderwidth=1,
                             relief="solid",
                             padding=7, # Slightly increased padding
                             font=self.font_normal)
        self.style.map("TEntry", 
                       bordercolor=[("focus", self.accent_color), ("!focus", self.border_color)])

    def create_ui(self):
        """Create the application's user interface with refined styling"""
        # Main container with increased padding
        self.main_frame = ttk.Frame(self.root, style="TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=35, pady=30) # Increased padx
        
        # --- Header section --- 
        header_frame = ttk.Frame(self.main_frame, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 30)) # Increased bottom padding
        header_frame.columnconfigure(1, weight=1) 

        # Logo Label
        if self.logo_image:
            logo_label = tk.Label(header_frame, image=self.logo_image, bg=self.bg_color)
            logo_label.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 20)) # Increased right padding

        # Frame for header text
        header_text_frame = ttk.Frame(header_frame, style="TFrame")
        header_text_frame.grid(row=0, column=1, rowspan=2, sticky="nsew") # Stick all directions

        header_label = ttk.Label(header_text_frame, 
                                text="Resume Analysis Engine", 
                                style="Header.TLabel")
        header_label.pack(anchor=tk.W, pady=(2,0)) # Add slight top padding
        
        description_label = ttk.Label(header_text_frame, 
                                     text="Add resumes individually and match candidates to job requirements.",
                                     style="TLabel",
                                     foreground=self.secondary_text_color)
        description_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Back button
        back_button = ttk.Button(header_frame, 
                                text="Back to Menu",
                                command=self.back_to_main,
                                style="TButton")
        back_button.grid(row=0, column=2, sticky="ne", padx=(10, 0), pady=(0, 5))
        
        # --- Content area --- 
        content_frame = ttk.Frame(self.main_frame, style="TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        content_frame.columnconfigure(0, weight=2) 
        content_frame.columnconfigure(1, weight=1) 
        content_frame.rowconfigure(0, weight=1)

        # --- Left panel --- 
        left_panel = ttk.Frame(content_frame, style="TFrame")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 25)) # Increased right padding
        left_panel.rowconfigure(1, weight=1) 
        left_panel.columnconfigure(0, weight=1)

        # Individual Resume Section (Card Style)
        upload_card = ttk.Frame(left_panel, style="Card.TFrame", padding=(25, 20)) # Increased padding
        upload_card.grid(row=0, column=0, sticky="ew", pady=(0, 25)) # Increased bottom padding
        
        upload_header = ttk.Label(upload_card, 
                                 text="Add Individual Resumes", 
                                 style="SubHeader.TLabel")
        upload_header.pack(anchor=tk.W, pady=(0, 5))
        
        upload_desc = ttk.Label(upload_card, 
                               text="Enter candidate details and select their resume in PDF format.",
                               style="Card.TLabel",
                               foreground=self.secondary_text_color)
        upload_desc.pack(anchor=tk.W, pady=(0, 18)) # Increased bottom padding
        
        # Candidate Name Input
        name_frame = ttk.Frame(upload_card, style="Card.TFrame")
        name_frame.pack(fill=tk.X, pady=(0, 10))
        
        name_label = ttk.Label(name_frame, text="Candidate Name:", 
                              style="Card.TLabel",
                              font=self.font_bold)
        name_label.pack(side=tk.LEFT, padx=(0, 10))
        
        name_entry = ttk.Entry(name_frame, textvariable=self.candidate_name, 
                              style="TEntry")
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # File selection
        file_frame = ttk.Frame(upload_card, style="Card.TFrame")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        select_button = ttk.Button(file_frame, 
                                  text="Select PDF",
                                  command=self.select_pdf,
                                  style="TButton")
        select_button.pack(side=tk.LEFT)
        
        self.file_label = ttk.Label(file_frame, 
                                   text="No file selected",
                                   style="Card.TLabel",
                                   foreground=self.secondary_text_color,
                                   font=self.font_small)
        self.file_label.pack(side=tk.LEFT, padx=(15, 0), pady=(5,0), anchor=tk.W)
        
        # Add Resume Button with counter
        add_frame = ttk.Frame(upload_card, style="Card.TFrame")
        add_frame.pack(fill=tk.X, pady=(5, 0))
        add_frame.columnconfigure(1, weight=1)
        
        add_button = ttk.Button(add_frame, 
                               text="Add Resume",
                               command=self.add_resume,
                               style="TButton")
        add_button.grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        # Resume counter
        self.counter_label = ttk.Label(add_frame, 
                                      text="Resumes added: 0",
                                      style="Card.TLabel",
                                      foreground=self.secondary_text_color,
                                      font=self.font_small)
        self.counter_label.grid(row=0, column=1, sticky="w")
        
        # Job details section (Card Style)
        job_card = ttk.Frame(left_panel, style="Card.TFrame", padding=(25, 20)) # Increased padding
        job_card.grid(row=1, column=0, sticky="nsew") 
        job_card.rowconfigure(4, weight=1) # Allow description text area to expand
        job_card.columnconfigure(0, weight=1)

        job_header = ttk.Label(job_card, 
                              text="Job Specification",
                              style="SubHeader.TLabel")
        job_header.grid(row=0, column=0, sticky="w", pady=(0, 20)) # Increased bottom padding
        
        # Job role input
        role_label = ttk.Label(job_card, 
                              text="Job Role", # Simplified label
                              style="Card.TLabel",
                              font=self.font_bold)
        role_label.grid(row=1, column=0, sticky="w", pady=(0, 6))
        
        role_entry = ttk.Entry(job_card, 
                              textvariable=self.job_role, 
                              style="TEntry")
        role_entry.grid(row=2, column=0, sticky="ew", pady=(0, 18)) # Increased bottom padding
        
        # Job description input
        desc_label = ttk.Label(job_card, 
                              text="Job Description", # Simplified label
                              style="Card.TLabel",
                              font=self.font_bold)
        desc_label.grid(row=3, column=0, sticky="w", pady=(0, 6))
        
        # --- ScrolledText wrapped in a styled Frame for border --- 
        text_frame = ttk.Frame(job_card, style="TextBorder.TFrame")
        text_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 18)) # Increased bottom padding
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        
        self.job_desc_text = ScrolledText(text_frame, height=10, 
                                        background=self.card_bg, 
                                        foreground=self.text_color,
                                        font=self.font_normal,
                                        borderwidth=0, # Border handled by frame
                                        relief="flat", # Border handled by frame
                                        padx=8, 
                                        pady=8,
                                        wrap=tk.WORD)
        self.job_desc_text.grid(row=0, column=0, sticky="nsew", padx=1, pady=1) # Add 1px padding inside border
        # --- End ScrolledText wrapper --- 
        
        # Process button and added resumes counter
        action_frame = ttk.Frame(job_card, style="Card.TFrame")
        action_frame.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        action_frame.columnconfigure(1, weight=1) 

        # Process button
        self.process_btn = ttk.Button(action_frame, 
                                     text="Analyze Resumes",
                                     command=self.process_resumes_thread,
                                     style="TButton",
                                     state=tk.DISABLED)
        self.process_btn.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 20))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(action_frame, 
                                      orient=tk.HORIZONTAL, 
                                      mode='determinate',
                                      variable=self.progress_var,
                                      style="TProgressbar")
        self.progress.grid(row=0, column=1, sticky="ew", pady=(0, 3), padx=(0, 5))
        
        # Progress label
        self.progress_label = ttk.Label(action_frame, 
                                      text="Add at least one resume to proceed",
                                      style="Card.TLabel",
                                      foreground=self.secondary_text_color,
                                      font=self.font_small)
        self.progress_label.grid(row=1, column=1, sticky="w")
        
        # --- Right panel --- 
        self.right_panel_card = ttk.Frame(content_frame, style="Card.TFrame", padding=(25, 20)) # Increased padding
        self.right_panel_card.grid(row=0, column=1, sticky="nsew")
        self.right_panel_card.rowconfigure(1, weight=1) 
        self.right_panel_card.columnconfigure(0, weight=1)

        # This header will change based on state (Added Resumes vs Rankings)
        self.results_header = ttk.Label(self.right_panel_card, 
                                     text="Added Resumes", 
                                     style="SubHeader.TLabel")
        self.results_header.grid(row=0, column=0, sticky="w", pady=(0, 20)) # Increased bottom padding
        
        # Scrollable results area
        self.results_outer_frame = ttk.Frame(self.right_panel_card, style="Card.TFrame")
        self.results_outer_frame.grid(row=1, column=0, sticky="nsew")
        self.results_outer_frame.rowconfigure(0, weight=1)
        self.results_outer_frame.columnconfigure(0, weight=1)
        
        self.result_canvas = tk.Canvas(self.results_outer_frame, bg=self.card_bg, highlightthickness=0)
        self.result_canvas.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(self.results_outer_frame, orient=tk.VERTICAL, 
                                 command=self.result_canvas.yview, style="Vertical.TScrollbar")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.result_canvas.configure(yscrollcommand=scrollbar.set)
        self.result_canvas.bind('<Configure>', 
                              lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all")))
        self.result_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Create a frame inside the canvas to hold the resume items
        self.resume_list_frame = ttk.Frame(self.result_canvas, style="Card.TFrame")
        self.result_canvas.create_window((0, 0), window=self.resume_list_frame, anchor=tk.NW)
        
        self.resume_list_frame.bind("<Configure>", 
                                  lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"), 
                                                                      width=e.width))
        
        # Initial empty state message
        self.empty_label = ttk.Label(self.resume_list_frame, 
                                   text="Added resumes will appear here.",
                                   style="Card.TLabel",
                                   foreground=self.secondary_text_color,
                                   justify=tk.CENTER)
        self.empty_label.pack(pady=30, padx=20)
        
        # --- Pagination Controls (hidden initially) --- 
        self.pagination_frame = ttk.Frame(self.right_panel_card, style="Card.TFrame")
        self.pagination_frame.grid(row=2, column=0, sticky="ew", pady=(15, 0)) # Increased top padding
        self.pagination_frame.columnconfigure(1, weight=1)
        
        self.prev_button = ttk.Button(self.pagination_frame,
                                     text="Previous",
                                     command=self.prev_page,
                                     style="Pagination.TButton",
                                     state=tk.DISABLED)
        self.prev_button.grid(row=0, column=0, sticky="w", padx=5)
        
        self.page_label = ttk.Label(self.pagination_frame, 
                                   text="Page 1 of 1", 
                                   style="Card.TLabel", 
                                   font=self.font_small,
                                   anchor=tk.CENTER)
        self.page_label.grid(row=0, column=1, sticky="ew")
        
        self.next_button = ttk.Button(self.pagination_frame,
                                     text="Next",
                                     command=self.next_page,
                                     style="Pagination.TButton",
                                     state=tk.DISABLED)
        self.next_button.grid(row=0, column=2, sticky="e", padx=5)
        
        # Hide pagination initially
        self.pagination_frame.grid_remove()

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling for the results canvas."""
        # Determine scroll direction and factor (adjust factor for sensitivity)
        scroll_factor = -1 * (event.delta // 120) 
        self.result_canvas.yview_scroll(scroll_factor, "units")

    def select_pdf(self):
        """Handle resume file selection"""
        file_path = filedialog.askopenfilename(
            title="Select Resume PDF",
            filetypes=[("PDF files", "*.pdf")]
        )
        if file_path:
            self.selected_file = file_path
            self.file_label.config(text=os.path.basename(file_path))
    
    def add_resume(self):
        """Add the current resume to the list"""
        name = self.candidate_name.get().strip()
        if not name:
            messagebox.showwarning("Input Required", "Please enter a candidate name.")
            return
            
        if not self.selected_file:
            messagebox.showwarning("File Required", "Please select a PDF resume file.")
            return
        
        try:
            # Extract text from PDF
            text = extract_text(self.selected_file)
            normalized_text = normalize_text(text)
            
            # Add to resume list
            resume_info = {
                "candidate_name": name,
                "full_path": self.selected_file,
                "file_name": os.path.basename(self.selected_file),
                "text": normalized_text  # Store extracted text
            }
            self.resume_list.append(resume_info)
            
            # Update counter
            self.counter_label.config(text=f"Resumes added: {len(self.resume_list)}")
            
            # Clear the empty label if this is the first resume
            if len(self.resume_list) == 1:
                self.empty_label.pack_forget()
            
            # Add to visual list in the right panel
            result_card = ttk.Frame(self.resume_list_frame, style="Card.TFrame", padding=(10, 8))
            result_card.pack(fill=tk.X, padx=5, pady=(0, 8))
            
            # Configure columns for Rank, Details, Preview
            result_card.columnconfigure(1, weight=1) # Details frame expands
            result_card.columnconfigure(2, weight=0) # Preview button fixed size
            
            # Index indicator (1, 2, 3, etc)
            index = len(self.resume_list)
            index_label = ttk.Label(result_card, 
                                    text=f"{index}", 
                                    font=(self.font_family, 10, "bold"), 
                                    foreground="white", 
                                    background=self.accent_color,
                                    padding=(4, 2),
                                    anchor=tk.CENTER,
                                    width=3)
            index_label.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky="ns")
            
            # Resume details Frame
            details_frame = ttk.Frame(result_card, style="Card.TFrame")
            details_frame.grid(row=0, column=1, rowspan=2, sticky="ew", padx=(0, 10))
            
            name_label = ttk.Label(details_frame, 
                                  text=name,
                                  font=self.font_bold,
                                  style="Card.TLabel")
            name_label.pack(anchor=tk.W)
            
            file_label = ttk.Label(details_frame, 
                                  text=resume_info['file_name'],
                                  style="Secondary.TLabel",
                                  font=self.font_small)
            file_label.pack(anchor=tk.W)
            
            # Preview Button
            preview_button = ttk.Button(result_card,
                                      text="Preview",
                                      style="Preview.TButton",
                                      command=lambda p=resume_info["full_path"]: self.preview_pdf(p))
            preview_button.grid(row=0, column=2, rowspan=2, sticky="e")
            
            # Enable the proceed button if we have at least one resume
            if len(self.resume_list) > 0:
                self.process_btn.config(state=tk.NORMAL)
            
            # Update canvas scroll region
            self.resume_list_frame.update_idletasks()
            self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
            
            # Clear inputs for next resume
            self.candidate_name.set("")
            self.selected_file = None
            self.file_label.config(text="No file selected")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the PDF file:\n{str(e)}")
    
    def preview_pdf(self, pdf_path):
        """Display a preview of the selected PDF"""
        try:
            # Create preview in the right panel
            # First remove any existing preview
            for widget in self.resume_list_frame.winfo_children():
                widget.destroy()
                
            # Create preview frame
            preview_frame = ttk.Frame(self.resume_list_frame, style="Card.TFrame")
            preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add the file name
            file_name = os.path.basename(pdf_path)
            name_label = ttk.Label(preview_frame, 
                                  text=f"Preview: {file_name}",
                                  style="Card.TLabel",
                                  font=self.font_bold)
            name_label.pack(anchor=tk.W, pady=(0, 10))
            
            try:
                # Try to render the first page of the PDF
                doc = fitz.open(pdf_path)
                if doc.page_count > 0:
                    page = doc[0]  # First page
                    
                    # Render at a reasonable size
                    zoom_matrix = fitz.Matrix(0.5, 0.5)  # 50% zoom
                    pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_tk = ImageTk.PhotoImage(image=img)
                    
                    # Keep a reference to prevent garbage collection
                    preview_frame.img_tk = img_tk
                    
                    # Display the image
                    img_label = ttk.Label(preview_frame)
                    img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    img_label.configure(image=img_tk)
                    
                    # Add page info
                    page_info = ttk.Label(preview_frame, 
                                         text=f"Page 1 of {doc.page_count}",
                                         style="Secondary.TLabel")
                    page_info.pack(pady=(5, 0))
                    
                    doc.close()
                    
            except Exception as e:
                # If can't render, show error message
                error_msg = f"Could not render PDF preview: {str(e)}"
                preview_text = ttk.Label(preview_frame, 
                                       text=error_msg,
                                       style="Card.TLabel",
                                       foreground="red",
                                       wraplength=300,
                                       justify=tk.CENTER)
                preview_text.pack(expand=True, pady=50)
            
            # Add a "Back to Resume List" button
            back_btn = ttk.Button(preview_frame, 
                                 text="Back to Resume List",
                                 command=self.refresh_resume_list,
                                 style="TButton")
            back_btn.pack(pady=(10, 0))
            
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not preview the file:\n{str(e)}")
            self.refresh_resume_list()
    
    def refresh_resume_list(self):
        """Refresh the resume list display"""
        # Clear existing content
        for widget in self.resume_list_frame.winfo_children():
            widget.destroy()
            
        if not self.resume_list:
            # Show empty state message
            self.empty_label = ttk.Label(self.resume_list_frame, 
                                       text="Added resumes will appear here.",
                                       style="Card.TLabel",
                                       foreground=self.secondary_text_color,
                                       justify=tk.CENTER)
            self.empty_label.pack(pady=30, padx=20)
            return
            
        # Recreate all resume items
        for i, resume_info in enumerate(self.resume_list):
            result_card = ttk.Frame(self.resume_list_frame, style="Card.TFrame", padding=(10, 8))
            result_card.pack(fill=tk.X, padx=5, pady=(0, 8))
            
            # Configure columns
            result_card.columnconfigure(1, weight=1)
            result_card.columnconfigure(2, weight=0)
            
            # Index indicator
            index = i + 1
            index_label = ttk.Label(result_card, 
                                   text=f"{index}", 
                                   font=(self.font_family, 10, "bold"), 
                                   foreground="white", 
                                   background=self.accent_color,
                                   padding=(4, 2),
                                   anchor=tk.CENTER,
                                   width=3)
            index_label.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky="ns")
            
            # Resume details
            details_frame = ttk.Frame(result_card, style="Card.TFrame")
            details_frame.grid(row=0, column=1, rowspan=2, sticky="ew", padx=(0, 10))
            
            name_label = ttk.Label(details_frame, 
                                  text=resume_info["candidate_name"],
                                  font=self.font_bold,
                                  style="Card.TLabel")
            name_label.pack(anchor=tk.W)
            
            file_label = ttk.Label(details_frame, 
                                  text=resume_info["file_name"],
                                  style="Secondary.TLabel",
                                  font=self.font_small)
            file_label.pack(anchor=tk.W)
            
            # Preview button
            preview_button = ttk.Button(result_card,
                                      text="Preview",
                                      style="Preview.TButton",
                                      command=lambda p=resume_info["full_path"]: self.preview_pdf(p))
            preview_button.grid(row=0, column=2, rowspan=2, sticky="e")
            
        # Update scroll region
        self.resume_list_frame.update_idletasks()
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))

    def back_to_main(self):
        """Return to the landing screen"""
        # Clean up current screen
        self.main_frame.destroy()
        
        # Go back to landing screen
        LandingScreen(self.root)
    
    def process_resumes_thread(self):
        """Start resume processing in a separate thread"""
        if not self.resume_list:
            messagebox.showwarning("Input Required", "Please add at least one resume first.")
            return
        
        role = self.job_role.get().strip()
        description = self.job_desc_text.get(1.0, tk.END).strip()
        
        if not role:
            messagebox.showwarning("Input Required", "Please enter the Job Role.")
            return
            
        if not description:
            messagebox.showwarning("Input Required", "Please enter the Job Description.")
            return
        
        # Change the right panel to display results
        self.results_header.config(text="Candidate Rankings")
        
        # Clear previous results
        for widget in self.resume_list_frame.winfo_children():
            widget.destroy()
            
        # Reset pagination state
        self.current_page = 1
        self.all_ranked_resumes = []
        self.update_pagination_controls()  # Update controls for empty/processing state
        
        # Show processing message
        processing_label = ttk.Label(self.resume_list_frame, 
                                    text="Analyzing...", 
                                    style="Card.TLabel",
                                    foreground=self.secondary_text_color,
                                    font=self.font_bold) 
        processing_label.pack(pady=30, padx=20)
        
        # Reset progress
        self.update_progress(0, "Initializing analysis...") 
        
        # Disable button during processing
        self.process_btn.config(state=tk.DISABLED)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=lambda: self.process_resumes_worker(role, description))
        thread.daemon = True
        thread.start()
        
    def update_progress(self, value, message):
        """Update the progress bar and label"""
        self.progress_var.set(value * 100)
        self.progress_label.config(text=message)
        self.root.update_idletasks()
        
    def process_resumes_worker(self, role, description):
        """Process resumes in a background thread"""
        try:
            # Load model 
            model = load_model(progress_callback=self.update_progress)
            
            # Use the pre-processed resume data
            resumes = self.resume_list
            self.update_progress(0.5, "Processing resumes...")
            
            if not resumes:
                raise ValueError("No valid resumes could be processed.")

            # Create full job description
            full_job_desc = f"Job Role: {role}\n\n{description}"
            
            # Ranking progress callback
            def ranking_progress_callback(value, message):
                self.update_progress(0.5 + value * 0.5, message)

            # Rank resumes
            ranked_resumes_data = rank_resumes(resumes, full_job_desc, model, progress_callback=ranking_progress_callback)
            
            # Update UI 
            self.root.after(0, lambda: self.update_results_display(ranked_resumes_data))
            
        except Exception as e:
            self.root.after(0, lambda: self.handle_error(str(e)))
            
    def update_results_display(self, ranked_resumes_data):
        """Handles the full ranked list and updates the display for the current page."""
        self.all_ranked_resumes = ranked_resumes_data  # Store all results
        self.current_page = 1  # Reset to page 1
        self.is_analyzed = True  # Mark as analyzed
        self.update_results_page()  # Display the first page
        
        # Show pagination controls
        self.pagination_frame.grid()

    def update_results_page(self):
        """Updates the results list to show items for the current page."""
        # Clear previous results from the list frame
        for widget in self.resume_list_frame.winfo_children():
            widget.destroy()

        # Calculate total pages
        self.total_pages = max(1, (len(self.all_ranked_resumes) + self.results_per_page - 1) // self.results_per_page)

        # Determine the slice of results for the current page
        start_index = (self.current_page - 1) * self.results_per_page
        end_index = start_index + self.results_per_page
        resumes_to_display = self.all_ranked_resumes[start_index:end_index]
        
        # Update progress label
        if self.all_ranked_resumes:
            self.update_progress(1.0, "Analysis complete.")
        else:
            self.update_progress(1.0, "No results found.")

        # Display results for the current page
        if not resumes_to_display and self.current_page == 1: 
            self.empty_label = ttk.Label(self.resume_list_frame, 
                                   text="No matching results found.",
                                   style="Card.TLabel",
                                   foreground=self.secondary_text_color)
            self.empty_label.pack(pady=30, padx=20)
        else:
            for i, (resume, score) in enumerate(resumes_to_display):
                rank = start_index + i + 1
                # Create a card for each result
                result_card = ttk.Frame(self.resume_list_frame, style="Card.TFrame", padding=(10, 8))
                result_card.pack(fill=tk.X, padx=5, pady=(0, 8))
                # Configure columns for Rank, Details, Preview, Score
                result_card.columnconfigure(1, weight=1)  # Details frame expands
                result_card.columnconfigure(2, weight=0)  # Preview button fixed size
                result_card.columnconfigure(3, weight=0)  # Score fixed size

                # Rank Indicator
                rank_label = ttk.Label(result_card, 
                                      text=f"{rank}", 
                                      font=(self.font_family, 10, "bold"), 
                                      foreground="white", 
                                      background=self.get_rank_color(score),
                                      padding=(4, 2),
                                      anchor=tk.CENTER,
                                      width=3)
                rank_label.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky="ns")

                # Resume details Frame
                details_frame = ttk.Frame(result_card, style="Card.TFrame")
                details_frame.grid(row=0, column=1, rowspan=2, sticky="ew", padx=(0, 10))
                
                name_label = ttk.Label(details_frame, 
                                      text=resume['candidate_name'],
                                      font=self.font_bold,
                                      style="Card.TLabel")
                name_label.pack(anchor=tk.W)
                
                file_label = ttk.Label(details_frame, 
                                      text=resume['file_name'],
                                      style="Secondary.TLabel",
                                      font=self.font_small)
                file_label.pack(anchor=tk.W)
                
                # Preview Button
                preview_button = ttk.Button(result_card,
                                          text="Preview",
                                          style="Preview.TButton",
                                          command=lambda p=resume["full_path"]: self.preview_pdf(p))
                preview_button.grid(row=0, column=2, rowspan=2, sticky="e", padx=(0,10))

                # Score display
                score_label = ttk.Label(result_card, 
                                       text=f"{score*100:.1f}% Match",
                                       font=self.font_bold,
                                       foreground=self.get_rank_color(score),
                                       style="Card.TLabel")
                score_label.grid(row=0, column=3, rowspan=2, padx=(0, 0), sticky="e")
        
        # Update pagination controls (buttons and label)
        self.update_pagination_controls()

        # Re-enable the main process button if it was disabled
        self.process_btn.config(state=tk.NORMAL)
        
        # Ensure the scroll region is updated and scroll to top
        self.result_canvas.update_idletasks()
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
        self.result_canvas.yview_moveto(0)  # Scroll to top when page changes
        
    def update_pagination_controls(self):
        """Updates the state and text of the pagination controls."""
        if not self.all_ranked_resumes:
            self.page_label.config(text="")
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
        else:
            self.page_label.config(text=f"Page {self.current_page} of {self.total_pages}")
            # Update Previous button state
            if self.current_page > 1:
                self.prev_button.config(state=tk.NORMAL)
            else:
                self.prev_button.config(state=tk.DISABLED)
            
            # Update Next button state
            if self.current_page < self.total_pages:
                self.next_button.config(state=tk.NORMAL)
            else:
                self.next_button.config(state=tk.DISABLED)
                
    def next_page(self):
        """Go to the next page of results."""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.update_results_page()

    def prev_page(self):
        """Go to the previous page of results."""
        if self.current_page > 1:
            self.current_page -= 1
            self.update_results_page()
            
    def get_rank_color(self, score):
        """Return a professional color based on the match score"""
        if score >= 0.75:
            return "#107C10"  # Dark Green (High match)
        elif score >= 0.60:
            return "#0078D4"  # Blue (Good match)
        elif score >= 0.45:
            return "#FCAA12"  # Amber (Medium match)
        else:
            return "#D83B01"  # Dark Orange/Red (Low match)
            
    def handle_error(self, error_msg):
        """Handle and display errors with professional styling"""
        # Update progress
        self.update_progress(0, "Error occurred.")
        
        # Display error message box
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n\n{error_msg}")
        
        # Clear results and reset pagination
        self.all_ranked_resumes = []
        self.current_page = 1
        self.update_pagination_controls()
        
        # Add error message to results area
        error_label = ttk.Label(self.resume_list_frame, 
                              text=f"Error: {error_msg}",
                              foreground="#D83B01",  # Use error color
                              style="Card.TLabel",
                              wraplength=300)  # Wrap long messages
        error_label.pack(pady=30, padx=20)
        
        # Re-enable the process button
        self.process_btn.config(state=tk.NORMAL)
        
        # Ensure the scroll region is updated
        self.result_canvas.update_idletasks()
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))

    def proceed_to_analysis(self):
        """Replaced by process_resumes_thread - this is no longer used"""
        pass

class EnhancedResumeRankerGUI:
    def __init__(self, root):
        self.root = root
        # --- Adjustments for seamless transition ---
        # Don't reset title/geometry if already set by LandingScreen
        # Assume root is already configured with bg_color
        if not hasattr(root, 'main_app_initialized'): # Check if already initialized
            self.root.title("Resume Analysis Engine")
            self.root.geometry("1100x750") 
            self.root.minsize(900, 650)
            root.main_app_initialized = True # Mark as initialized
        # --- End Adjustments ---

        # Set the application theme (reusing global settings)
        self.set_theme()
        
        # Variables to store inputs
        self.selected_files = []
        self.job_role = tk.StringVar()
        
        # Storage for pre-loaded resume data (if coming from IndividualResumeScreen)
        self.preloaded_resume_data = None
        
        # Pagination state
        self.results_per_page = 10
        self.current_page = 1
        self.total_pages = 1
        self.all_ranked_resumes = [] # To store all results

        # Load Logo
        self.logo_image = None
        try:
            # Determine base path whether running as script or frozen executable
            if getattr(sys, 'frozen', False):
                # If the application is run as a bundle/executable, the base path is the sys._MEIPASS
                base_path = sys._MEIPASS
            else:
                # If run as a script, the base path is the script directory
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            # Define the logo path relative to the base path
            # Assumes the logo will be in the root directory of the bundle/script location
            logo_path = os.path.join(base_path, "public/images", "ATOMS_LOGO.png")
            
            # Use the corrected absolute path provided by the user - REMOVED OLD PATH
            # logo_path = r"C:\Users\502ch\connected-website\resume-matcher\public\images\ATOMS_LOGO.png"
            
            if os.path.exists(logo_path):
                img = Image.open(logo_path)
                # Resize logo to a suitable size (e.g., height 60 pixels)
                img_height = 60
                img_ratio = img.height / img_height
                img_width = int(img.width / img_ratio)
                img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                self.logo_image = ImageTk.PhotoImage(img)
            else:
                print(f"Warning: Logo file not found at expected location: {logo_path}")
        except Exception as e:
            print(f"Error loading logo: {e}")

        # Create the UI components
        self.create_ui()
    
    def set_theme(self):
        """Set the application color scheme and styling for a professional look"""
        # Use global theme settings
        self.bg_color = bg_color
        self.accent_color = accent_color
        self.accent_hover_color = accent_hover_color
        self.text_color = text_color
        self.secondary_text_color = "#595959" # Can remain specific if needed
        self.card_bg = card_bg
        self.border_color = border_color
        
        self.font_family = font_family
        self.font_normal = font_normal
        self.font_bold = font_bold
        self.font_small = (self.font_family, 9) # Can remain specific
        self.font_large_bold = font_large_bold
        self.font_medium_bold = font_medium_bold

        # Configure root window (ensure bg color is set, though likely already done)
        self.root.configure(bg=self.bg_color)
        
        # Define styles for ttk widgets
        self.style = ttk.Style()
        self.style.theme_use('clam') # Use a theme that allows more customization

        # General Frame Style
        self.style.configure("TFrame", background=self.bg_color)
        
        # Card Style Frame
        self.style.configure("Card.TFrame", 
                             background=self.card_bg,
                             borderwidth=1,
                             relief="solid")
        self.style.map("Card.TFrame", bordercolor=[("!focus", self.border_color)])
        
        # Frame for ScrolledText border
        self.style.configure("TextBorder.TFrame", 
                             background=self.card_bg, # Match card background
                             borderwidth=1, 
                             relief="solid")
        self.style.map("TextBorder.TFrame", bordercolor=[("!focus", self.border_color)])

        # Label Styles
        self.style.configure("TLabel", 
                             background=self.bg_color, 
                             foreground=self.text_color,
                             font=self.font_normal)
        
        self.style.configure("Card.TLabel", 
                             background=self.card_bg, 
                             foreground=self.text_color,
                             font=self.font_normal)

        self.style.configure("Header.TLabel", 
                             background=self.bg_color, 
                             foreground=self.text_color,
                             font=self.font_large_bold)
        
        self.style.configure("SubHeader.TLabel", 
                             background=self.card_bg, 
                             foreground=self.text_color,
                             font=self.font_medium_bold)
        
        self.style.configure("Secondary.TLabel", 
                             background=self.card_bg, 
                             foreground=self.secondary_text_color,
                             font=self.font_small)

        # Button Style
        self.style.configure("TButton", 
                             font=self.font_bold,
                             background=self.accent_color,
                             foreground="white",
                             borderwidth=0,
                             padding=(14, 9), # Slightly increased padding
                             relief="flat",
                             cursor="hand2") # Add cursor directly here 
        self.style.map("TButton",
                       background=[('active', self.accent_hover_color), 
                                   ('disabled', '#B0B0B0'), # Explicit disabled background
                                   ('!disabled', self.accent_color)],
                       foreground=[('active', 'white'), ('disabled', '#F0F0F0')])
        
        # Pagination Button Style (smaller)
        self.style.configure("Pagination.TButton", 
                             font=(self.font_family, 9, "bold"),
                             background=self.accent_color,
                             foreground="white",
                             borderwidth=0,
                             padding=(8, 5), # Smaller padding
                             relief="flat",
                             cursor="hand2")
        self.style.map("Pagination.TButton",
                       background=[('active', self.accent_hover_color), 
                                   ('disabled', '#B0B0B0'), 
                                   ('!disabled', self.accent_color)],
                       foreground=[('active', 'white'), ('disabled', '#F0F0F0')])

        # Entry Style
        self.style.configure("TEntry", 
                             fieldbackground="white",
                             borderwidth=1,
                             relief="solid",
                             padding=7, # Slightly increased padding
                             font=self.font_normal)
        self.style.map("TEntry", 
                       bordercolor=[("focus", self.accent_color), ("!focus", self.border_color)])

        # Progressbar Style
        self.style.configure("TProgressbar", 
                             thickness=15, 
                             background=self.accent_color,
                             troughcolor=self.border_color,
                             borderwidth=0,
                             relief="flat")
        
        # Scrollbar Style
        self.style.configure("Vertical.TScrollbar", 
                             troughcolor=self.bg_color, 
                             background=self.accent_color, 
                             gripcount=0,
                             relief="flat",
                             arrowsize=14)
        self.style.map("Vertical.TScrollbar",
                       background=[('active', self.accent_hover_color)])

        # Add Preview Button Style
        self.style.configure("Preview.TButton",
                             font=(self.font_family, 8), # Smaller font
                             padding=(5, 3), # Smaller padding
                             relief="flat",
                             cursor="hand2")
        # Inherit colors from TButton but allow overrides if needed
        self.style.map("Preview.TButton",
                       background=[('active', self.accent_hover_color),
                                   ('!disabled', self.accent_color)],
                       foreground=[('active', 'white'), ('!disabled', 'white')])

    def create_ui(self):
        """Create the application's user interface with refined styling"""
        # Main container with increased padding
        self.main_frame = ttk.Frame(self.root, style="TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=35, pady=30) # Increased padx
        
        # --- Header section --- 
        header_frame = ttk.Frame(self.main_frame, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 30)) # Increased bottom padding
        header_frame.columnconfigure(1, weight=1) 

        # Logo Label
        if self.logo_image:
            logo_label = tk.Label(header_frame, image=self.logo_image, bg=self.bg_color)
            logo_label.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 20)) # Increased right padding

        # Frame for header text
        header_text_frame = ttk.Frame(header_frame, style="TFrame")
        header_text_frame.grid(row=0, column=1, rowspan=2, sticky="nsew") # Stick all directions

        header_label = ttk.Label(header_text_frame, 
                                text="Resume Analysis Engine", 
                                style="Header.TLabel")
        header_label.pack(anchor=tk.W, pady=(2,0)) # Add slight top padding
        
        description_label = ttk.Label(header_text_frame, 
                                     text="Efficiently match candidates to job requirements using advanced AI.",
                                     style="TLabel",
                                     foreground=self.secondary_text_color)
        description_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Back button
        back_button = ttk.Button(header_frame, 
                                text="Back to Menu",
                                command=self.back_to_landing,
                                style="TButton")
        back_button.grid(row=0, column=2, sticky="ne", padx=(10, 0), pady=(0, 5))
        
        # --- Content area --- 
        content_frame = ttk.Frame(self.main_frame, style="TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        content_frame.columnconfigure(0, weight=2) 
        content_frame.columnconfigure(1, weight=1) 
        content_frame.rowconfigure(0, weight=1)

        # --- Left panel --- 
        left_panel = ttk.Frame(content_frame, style="TFrame")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 25)) # Increased right padding
        left_panel.rowconfigure(1, weight=1) 
        left_panel.columnconfigure(0, weight=1)

        # Resume upload section (Card Style)
        upload_card = ttk.Frame(left_panel, style="Card.TFrame", padding=(25, 20)) # Increased padding
        upload_card.grid(row=0, column=0, sticky="ew", pady=(0, 25)) # Increased bottom padding
        
        upload_header = ttk.Label(upload_card, 
                                 text="Upload Resumes", 
                                 style="SubHeader.TLabel")
        upload_header.pack(anchor=tk.W, pady=(0, 5))
        
        upload_desc = ttk.Label(upload_card, 
                               text="Select one or more candidate resumes in PDF format.",
                               style="Card.TLabel",
                               foreground=self.secondary_text_color)
        upload_desc.pack(anchor=tk.W, pady=(0, 18)) # Increased bottom padding
        
        # Button Frame
        button_frame = ttk.Frame(upload_card, style="Card.TFrame")
        button_frame.pack(fill=tk.X)

        self.upload_btn = ttk.Button(button_frame, 
                                      text="Select Files",
                                      command=self.select_resumes,
                                      style="TButton")
        self.upload_btn.pack(side=tk.LEFT)
        
        self.files_label = ttk.Label(button_frame, 
                                    text="No files selected",
                                    style="Card.TLabel",
                                    foreground=self.secondary_text_color,
                                    font=self.font_small)
        self.files_label.pack(side=tk.LEFT, padx=(15, 0), pady=(5,0), anchor=tk.W)
        
        # Job details section (Card Style)
        job_card = ttk.Frame(left_panel, style="Card.TFrame", padding=(25, 20)) # Increased padding
        job_card.grid(row=1, column=0, sticky="nsew") 
        job_card.rowconfigure(4, weight=1) # Allow description text area to expand
        job_card.columnconfigure(0, weight=1)

        job_header = ttk.Label(job_card, 
                              text="Job Specification",
                              style="SubHeader.TLabel")
        job_header.grid(row=0, column=0, sticky="w", pady=(0, 20)) # Increased bottom padding
        
        # Job role input
        role_label = ttk.Label(job_card, 
                              text="Job Role", # Simplified label
                              style="Card.TLabel",
                              font=self.font_bold)
        role_label.grid(row=1, column=0, sticky="w", pady=(0, 6))
        
        role_entry = ttk.Entry(job_card, 
                              textvariable=self.job_role, 
                              style="TEntry")
        role_entry.grid(row=2, column=0, sticky="ew", pady=(0, 18)) # Increased bottom padding
        
        # Job description input
        desc_label = ttk.Label(job_card, 
                              text="Job Description", # Simplified label
                              style="Card.TLabel",
                              font=self.font_bold)
        desc_label.grid(row=3, column=0, sticky="w", pady=(0, 6))
        
        # --- ScrolledText wrapped in a styled Frame for border --- 
        text_frame = ttk.Frame(job_card, style="TextBorder.TFrame")
        text_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 18)) # Increased bottom padding
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        
        self.job_desc_text = ScrolledText(text_frame, height=10, 
                                        background=self.card_bg, 
                                        foreground=self.text_color,
                                        font=self.font_normal,
                                        borderwidth=0, # Border handled by frame
                                        relief="flat", # Border handled by frame
                                        padx=8, 
                                        pady=8,
                                        wrap=tk.WORD)
        self.job_desc_text.grid(row=0, column=0, sticky="nsew", padx=1, pady=1) # Add 1px padding inside border
        # --- End ScrolledText wrapper --- 
        
        # Process button and added resumes counter
        action_frame = ttk.Frame(job_card, style="Card.TFrame")
        action_frame.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        action_frame.columnconfigure(1, weight=1) 

        # Process button
        self.process_btn = ttk.Button(action_frame, 
                                     text="Analyze Resumes",
                                     command=self.process_resumes_thread,
                                     style="TButton")
        self.process_btn.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 20))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(action_frame, 
                                       orient=tk.HORIZONTAL, 
                                       mode='determinate',
                                       variable=self.progress_var,
                                       style="TProgressbar")
        self.progress.grid(row=0, column=1, sticky="ew", pady=(0, 3), padx=(0,5))
        
        # Progress label
        self.progress_label = ttk.Label(action_frame, 
                                      text="Ready",
                                      style="Card.TLabel",
                                      foreground=self.secondary_text_color,
                                      font=self.font_small)
        self.progress_label.grid(row=1, column=1, sticky="w")
        
        # --- Right panel --- 
        right_panel_card = ttk.Frame(content_frame, style="Card.TFrame", padding=(25, 20)) # Increased padding
        right_panel_card.grid(row=0, column=1, sticky="nsew")
        right_panel_card.rowconfigure(1, weight=1) 
        right_panel_card.columnconfigure(0, weight=1)

        results_header = ttk.Label(right_panel_card, 
                                  text="Added Resumes", 
                                  style="SubHeader.TLabel")
        results_header.grid(row=0, column=0, sticky="w", pady=(0, 20)) # Increased bottom padding
        
        # Scrollable results area
        self.results_outer_frame = ttk.Frame(right_panel_card, style="Card.TFrame")
        self.results_outer_frame.grid(row=1, column=0, sticky="nsew")
        self.results_outer_frame.rowconfigure(0, weight=1)
        self.results_outer_frame.columnconfigure(0, weight=1)
        
        self.result_canvas = tk.Canvas(self.results_outer_frame, bg=self.card_bg, highlightthickness=0)
        self.result_canvas.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(self.results_outer_frame, orient=tk.VERTICAL, command=self.result_canvas.yview, style="Vertical.TScrollbar")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.result_canvas.configure(yscrollcommand=scrollbar.set)
        self.result_canvas.bind('<Configure>', lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all")))
        self.result_canvas.bind_all("<MouseWheel>", self._on_mousewheel) 

        self.results_list_frame = ttk.Frame(self.result_canvas, style="Card.TFrame")
        self.result_canvas.create_window((0, 0), window=self.results_list_frame, anchor=tk.NW)
        
        self.results_list_frame.bind("<Configure>", 
                                  lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"), 
                                                                      width=e.width))
        
        # Initial empty state message
        self.empty_label = ttk.Label(self.results_list_frame, 
                                   text="Added resumes will appear here.",
                                   style="Card.TLabel",
                                   foreground=self.secondary_text_color,
                                   justify=tk.CENTER)
        self.empty_label.pack(pady=30, padx=20)

        # --- Pagination Controls --- 
        self.pagination_frame = ttk.Frame(right_panel_card, style="Card.TFrame")
        self.pagination_frame.grid(row=2, column=0, sticky="ew", pady=(15, 0)) # Increased top padding
        self.pagination_frame.columnconfigure(1, weight=1) 

        self.prev_button = ttk.Button(self.pagination_frame,
                                      text="Previous",
                                      command=self.prev_page,
                                      style="Pagination.TButton",
                                      state=tk.DISABLED)
        self.prev_button.grid(row=0, column=0, sticky="w", padx=5)

        self.page_label = ttk.Label(self.pagination_frame, 
                                    text="Page 1 of 1", 
                                    style="Card.TLabel", 
                                    font=self.font_small,
                                    anchor=tk.CENTER)
        self.page_label.grid(row=0, column=1, sticky="ew")

        self.next_button = ttk.Button(self.pagination_frame,
                                      text="Next",
                                      command=self.next_page,
                                      style="Pagination.TButton",
                                      state=tk.DISABLED)
        self.next_button.grid(row=0, column=2, sticky="e", padx=5)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling for the results canvas."""
        # Determine scroll direction and factor (adjust factor for sensitivity)
        scroll_factor = -1 * (event.delta // 120) 
        self.result_canvas.yview_scroll(scroll_factor, "units")

    def select_resumes(self):
        """Handle resume file selection"""
        files = filedialog.askopenfilenames(
            title="Select Resume PDFs",
            filetypes=[("PDF files", "*.pdf")]
        )
        if files:  # Only update if files were selected
            self.selected_files.clear()  # Clear previous selections
            self.selected_files.extend(files)
            self.files_label.config(text=f"{len(self.selected_files)} file(s) selected") # Updated text format
            
    def update_progress(self, value, message):
        """Update the progress bar and label"""
        self.progress_var.set(value * 100)
        self.progress_label.config(text=message)
        self.root.update_idletasks()

    def process_resumes_thread(self):
        """Start resume processing in a separate thread"""
        if not self.selected_files:
            messagebox.showwarning("Input Required", "Please select resume files first.") # Updated message
            return
        
        role = self.job_role.get()
        description = self.job_desc_text.get(1.0, tk.END)
        
        if not role.strip():
            messagebox.showwarning("Input Required", "Please enter the Job Role.") # Updated message
            return
            
        if not description.strip() or description.strip() == "":
            messagebox.showwarning("Input Required", "Please enter the Job Description.") # Updated message
            return
        
        # Clear previous results & show processing state
        for widget in self.results_list_frame.winfo_children():
            widget.destroy()
        
        # Reset pagination state
        self.current_page = 1
        self.all_ranked_resumes = []
        self.update_pagination_controls() # Update controls for empty/processing state

        processing_label = ttk.Label(self.results_list_frame, 
                                    text="Analyzing...", 
                                    style="Card.TLabel",
                                    foreground=self.secondary_text_color,
                                    font=self.font_bold) 
        processing_label.pack(pady=30, padx=20)
        
        # Reset progress
        self.update_progress(0, "Initializing analysis...") 
        
        # Disable button during processing
        self.process_btn.config(state=tk.DISABLED)
        self.style.map("TButton",
                       background=[('disabled', '#B0B0B0'), ('!disabled', self.accent_color)])
        
        # Start processing in a separate thread
        thread = threading.Thread(target=lambda: self.process_resumes_worker(role, description))
        thread.daemon = True
        thread.start()

    def process_resumes_worker(self, role, description):
        """Process resumes in a background thread"""
        try:
            # Load model 
            model = load_model(progress_callback=self.update_progress)
            
            # Use pre-processed resumes if available, otherwise process each resume
            if self.preloaded_resume_data:
                # Use pre-processed data
                resumes = self.preloaded_resume_data
                self.update_progress(0.5, "Using pre-processed resume data...")
            else:
                # Process each resume
                resumes = []
                total_files = len(self.selected_files)
                
                self.update_progress(0, f"Extracting text (0/{total_files})...")
                
                for i, pdf_path in enumerate(self.selected_files):
                    self.update_progress((i+1)/total_files * 0.5, f"Extracting text ({i+1}/{total_files}): {os.path.basename(pdf_path)}")
                    try:
                        text = extract_text(pdf_path)
                        candidate_name = extract_candidate_name(text)
                        normalized_text = normalize_text(text)
                        
                        resumes.append({
                            "file_name": os.path.basename(pdf_path),
                            "candidate_name": candidate_name if candidate_name else "Unknown Name",
                            "text": normalized_text,
                            "full_path": pdf_path # Store the full path
                        })
                    except Exception as e:
                        print(f"Warning: Could not process {os.path.basename(pdf_path)}: {e}")
            
            if not resumes:
                 raise ValueError("No valid resumes could be processed.")

            # Create full job description
            full_job_desc = f"Job Role: {role}\n\n{description}"
            
            # Ranking progress callback
            def ranking_progress_callback(value, message):
                 self.update_progress(0.5 + value * 0.5, message)

            # Rank resumes
            ranked_resumes_data = rank_resumes(resumes, full_job_desc, model, progress_callback=ranking_progress_callback)
            
            # Update UI 
            self.root.after(0, lambda: self.update_results_display(ranked_resumes_data))
            
        except Exception as e:
            self.root.after(0, lambda: self.handle_error(str(e)))

    def update_results_display(self, ranked_resumes_data):
        """Handles the full ranked list and updates the display for the current page."""
        self.all_ranked_resumes = ranked_resumes_data # Store all results
        self.current_page = 1 # Reset to page 1
        self.update_results_page() # Display the first page

    def update_results_page(self):
        """Updates the results list to show items for the current page."""
        # Clear previous results from the list frame
        for widget in self.results_list_frame.winfo_children():
            widget.destroy()

        # Calculate total pages
        self.total_pages = max(1, (len(self.all_ranked_resumes) + self.results_per_page - 1) // self.results_per_page)

        # Determine the slice of results for the current page
        start_index = (self.current_page - 1) * self.results_per_page
        end_index = start_index + self.results_per_page
        resumes_to_display = self.all_ranked_resumes[start_index:end_index]
        
        # Update progress label
        if self.all_ranked_resumes:
            self.update_progress(1.0, "Analysis complete.")
        else:
            self.update_progress(1.0, "No results found.")

        # Display results for the current page
        if not resumes_to_display and self.current_page == 1: 
            self.empty_label = ttk.Label(self.results_list_frame, 
                                   text="No matching results found.",
                                   style="Card.TLabel",
                                   foreground=self.secondary_text_color)
            self.empty_label.pack(pady=30, padx=20)
        else:
            for i, (resume, score) in enumerate(resumes_to_display):
                rank = start_index + i + 1
                # Create a card for each result
                result_card = ttk.Frame(self.results_list_frame, style="Card.TFrame", padding=(10, 8))
                result_card.pack(fill=tk.X, padx=5, pady=(0, 8))
                # Configure columns for Rank, Details, Preview, Score
                result_card.columnconfigure(1, weight=1) # Details frame expands
                result_card.columnconfigure(2, weight=0) # Preview button fixed size
                result_card.columnconfigure(3, weight=0) # Score fixed size

                # Rank Indicator
                rank_label = ttk.Label(result_card, 
                                       text=f"{rank}", 
                                       font=(self.font_family, 10, "bold"), 
                                       foreground="white", 
                                       background=self.get_rank_color(score),
                                       padding=(4, 2),
                                       anchor=tk.CENTER,
                                       width=3)
                rank_label.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky="ns")

                # Resume details Frame
                details_frame = ttk.Frame(result_card, style="Card.TFrame")
                details_frame.grid(row=0, column=1, rowspan=2, sticky="ew", padx=(0, 10))
                
                name_label = ttk.Label(details_frame, 
                                      text=resume['candidate_name'],
                                      font=self.font_bold,
                                      style="Card.TLabel")
                name_label.pack(anchor=tk.W)
                
                file_label = ttk.Label(details_frame, 
                                      text=resume['file_name'],
                                      style="Secondary.TLabel",
                                      font=self.font_small)
                file_label.pack(anchor=tk.W)
                
                # Preview Button
                preview_button = ttk.Button(result_card,
                                           text="Preview",
                                           style="Preview.TButton",
                                           command=lambda p=resume["full_path"]: self.preview_pdf(p))
                preview_button.grid(row=0, column=2, rowspan=2, sticky="e")

                # Score display
                score_label = ttk.Label(result_card, 
                                       text=f"{score*100:.1f}% Match",
                                       font=self.font_bold,
                                       foreground=self.get_rank_color(score),
                                       style="Card.TLabel")
                score_label.grid(row=0, column=3, rowspan=2, padx=(0, 0), sticky="e")
        
        # Update pagination controls (buttons and label)
        self.update_pagination_controls()

        # Re-enable the main process button if it was disabled
        self.process_btn.config(state=tk.NORMAL)
        self.style.map("TButton",
                       background=[('active', self.accent_hover_color), 
                                   ('!disabled', self.accent_color)],
                       foreground=[('active', 'white')])
                       
        # Ensure the scroll region is updated and scroll to top
        self.result_canvas.update_idletasks()
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
        self.result_canvas.yview_moveto(0) # Scroll to top when page changes

    def update_pagination_controls(self):
        """Updates the state and text of the pagination controls."""
        if not self.all_ranked_resumes:
            self.page_label.config(text="")
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
        else:
            self.page_label.config(text=f"Page {self.current_page} of {self.total_pages}")
            # Update Previous button state
            if self.current_page > 1:
                self.prev_button.config(state=tk.NORMAL)
            else:
                self.prev_button.config(state=tk.DISABLED)
            
            # Update Next button state
            if self.current_page < self.total_pages:
                self.next_button.config(state=tk.NORMAL)
            else:
                self.next_button.config(state=tk.DISABLED)

    def next_page(self):
        """Go to the next page of results."""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.update_results_page()

    def prev_page(self):
        """Go to the previous page of results."""
        if self.current_page > 1:
            self.current_page -= 1
            self.update_results_page()

    def get_rank_color(self, score):
        """Return a professional color based on the match score"""
        if score >= 0.75:
            return "#107C10"  # Dark Green (High match)
        elif score >= 0.60:
            return "#0078D4"  # Blue (Good match)
        elif score >= 0.45:
            return "#FCAA12"  # Amber (Medium match)
        else:
            return "#D83B01"  # Dark Orange/Red (Low match)

    def preview_pdf(self, pdf_path):
        """Display a preview of the selected PDF"""
        try:
            # Create preview in the right panel
            # First remove any existing preview
            for widget in self.results_list_frame.winfo_children():
                widget.destroy()
                
            # Create preview frame
            preview_frame = ttk.Frame(self.results_list_frame, style="Card.TFrame")
            preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add the file name
            file_name = os.path.basename(pdf_path)
            name_label = ttk.Label(preview_frame, 
                                  text=f"Preview: {file_name}",
                                  style="Card.TLabel",
                                  font=self.font_bold)
            name_label.pack(anchor=tk.W, pady=(0, 10))
            
            try:
                # Try to render the first page of the PDF
                doc = fitz.open(pdf_path)
                if doc.page_count > 0:
                    page = doc[0]  # First page
                    
                    # Render at a reasonable size
                    zoom_matrix = fitz.Matrix(0.5, 0.5)  # 50% zoom
                    pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_tk = ImageTk.PhotoImage(image=img)
                    
                    # Keep a reference to prevent garbage collection
                    preview_frame.img_tk = img_tk
                    
                    # Display the image
                    img_label = ttk.Label(preview_frame)
                    img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                    img_label.configure(image=img_tk)
                    
                    # Add page info
                    page_info = ttk.Label(preview_frame, 
                                         text=f"Page 1 of {doc.page_count}",
                                         style="Secondary.TLabel")
                    page_info.pack(pady=(5, 0))
                    
                    doc.close()
                    
            except Exception as e:
                # If can't render, show error message
                error_msg = f"Could not render PDF preview: {str(e)}"
                preview_text = ttk.Label(preview_frame, 
                                       text=error_msg,
                                       style="Card.TLabel",
                                       foreground="red",
                                       wraplength=300,
                                       justify=tk.CENTER)
                preview_text.pack(expand=True, pady=50)
            
            # Add a "Back to Results" button
            back_btn = ttk.Button(preview_frame, 
                                 text="Back to Results",
                                 command=self.refresh_results,
                                 style="TButton")
            back_btn.pack(pady=(10, 0))
            
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not preview the file:\n{str(e)}")
            self.refresh_results()
            
    def refresh_results(self):
        """Refresh the results display"""
        # Retrieve the current display state
        if self.all_ranked_resumes:
            # We have results, so show them
            self.update_results_page()
        else:
            # No results, show empty state
            for widget in self.results_list_frame.winfo_children():
                widget.destroy()
                
            self.empty_label = ttk.Label(self.results_list_frame, 
                                      text="Results will appear here after analysis.",
                                      style="Card.TLabel",
                                      foreground=self.secondary_text_color,
                                      justify=tk.CENTER)
            self.empty_label.pack(pady=30, padx=20)

    def handle_error(self, error_msg):
        """Handle and display errors with professional styling"""
        # Update progress
        self.update_progress(0, "Error occurred.") # Updated text
        
        # Display error message box
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n\n{error_msg}") # More specific title
        
        # Clear results and reset pagination
        self.all_ranked_resumes = []
        self.current_page = 1
        self.update_pagination_controls()
        
        # Add error message to results area
        error_label = ttk.Label(self.results_list_frame, 
                               text=f"Error: {error_msg}",
                               foreground="#D83B01", # Use error color
                               style="Card.TLabel",
                               wraplength=300) # Wrap long messages
        error_label.pack(pady=30, padx=20)
        
        # Re-enable the process button and reset style mapping
        self.process_btn.config(state=tk.NORMAL)
        self.style.map("TButton",
                       background=[('active', self.accent_hover_color), 
                                   ('!disabled', self.accent_color)],
                       foreground=[('active', 'white')])
        # Ensure the scroll region is updated
        self.result_canvas.update_idletasks()
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))

    def back_to_landing(self):
        """Return to the landing screen"""
        # Clean up current UI elements
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Re-initialize the landing screen
        LandingScreen(self.root)

def main():
    global app_root # Use the global root reference
    app_root = tk.Tk()
    
    # Apply initial theme settings to the root window
    app_root.configure(bg=bg_color) 
    style = ttk.Style(app_root)
    style.theme_use('clam') # Ensure theme is set early

    # Start with the Landing Screen
    LandingScreen(app_root)
    
    app_root.mainloop()

if __name__ == "__main__":
    main() 