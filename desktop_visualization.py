import sys
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QComboBox, QLineEdit,
                            QPushButton, QFrame, QTextEdit)
from PyQt5.QtCore import Qt
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore

class ResumeVisualizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATOMS Resume Ranking - 3D Visualization")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Score range slider
        score_frame = QFrame()
        score_frame.setFrameStyle(QFrame.StyledPanel)
        score_layout = QVBoxLayout(score_frame)
        score_label = QLabel("Score Range:")
        self.score_slider = QSlider(Qt.Horizontal)
        self.score_slider.setMinimum(0)
        self.score_slider.setMaximum(100)
        self.score_slider.setValue(0)  # Start with no filtering
        self.score_slider.valueChanged.connect(self.update_visualization)
        score_layout.addWidget(score_label)
        score_layout.addWidget(self.score_slider)
        controls_layout.addWidget(score_frame)
        
        # Color by dropdown
        color_frame = QFrame()
        color_frame.setFrameStyle(QFrame.StyledPanel)
        color_layout = QVBoxLayout(color_frame)
        color_label = QLabel("Color By:")
        self.color_combo = QComboBox()
        self.color_combo.addItems(['Transformer Score', 'TFIDF Score', 'Section Score', 'Combined Score'])
        self.color_combo.currentTextChanged.connect(self.update_visualization)
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_combo)
        controls_layout.addWidget(color_frame)
        
        # Search box
        search_frame = QFrame()
        search_frame.setFrameStyle(QFrame.StyledPanel)
        search_layout = QVBoxLayout(search_frame)
        search_label = QLabel("Search Candidate:")
        self.search_input = QLineEdit()
        self.search_input.textChanged.connect(self.update_visualization)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        controls_layout.addWidget(search_frame)
        
        layout.addLayout(controls_layout)
        
        # Create split view for 3D visualization and info
        split_layout = QHBoxLayout()
        
        # Create 3D view
        self.view = gl.GLViewWidget()
        split_layout.addWidget(self.view, stretch=2)
        
        # Create info panel
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.StyledPanel)
        info_layout = QVBoxLayout(info_frame)
        
        # Add debug info
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(QLabel("Debug Information:"))
        info_layout.addWidget(self.info_text)
        
        split_layout.addWidget(info_frame, stretch=1)
        layout.addLayout(split_layout)
        
        # Add grid at the bottom
        grid = gl.GLGridItem()
        grid.setSize(x=100, y=100, z=0)  # Make grid flat (z=0)
        grid.setSpacing(x=10, y=10, z=1)
        grid.translate(50, 50, 0)  # Position at bottom
        self.view.addItem(grid)
        
        # Set up camera with better initial position
        self.view.setCameraPosition(distance=200, elevation=30, azimuth=45)
        
        # Initialize scatter plot with larger points
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=15)
        self.view.addItem(self.scatter)
        
        # Load data after UI is set up
        self.load_data()
        
        # Update visualization
        self.update_visualization()
        
    def load_data(self):
        try:
            with open('hybrid_matching_results.json', 'r') as f:
                self.data = json.load(f)
            
            # Convert scores to percentages and ensure they're within bounds
            for item in self.data:
                item['transformer_score'] = min(100, max(0, item['transformer_score'] * 100))
                item['tfidf_score'] = min(100, max(0, item['tfidf_score'] * 100))
                item['section_score'] = min(100, max(0, item['section_score'] * 100))
                item['combined_score'] = min(100, max(0, item['combined_score'] * 100))
                # Truncate names to 20 characters
                item['candidate_name'] = item['candidate_name'][:20] + '...' if len(item['candidate_name']) > 20 else item['candidate_name']
            
            self.info_text.append(f"Loaded {len(self.data)} candidates successfully")
        except Exception as e:
            self.info_text.append(f"Error loading data: {str(e)}")
            self.data = []
    
    def update_visualization(self):
        if not self.data:
            self.info_text.append("No data available to visualize")
            return
        
        # Filter data based on score range
        score_threshold = self.score_slider.value()
        filtered_data = [item for item in self.data if item['combined_score'] >= score_threshold]
        
        # Filter by search term
        search_term = self.search_input.text().lower()
        if search_term:
            filtered_data = [item for item in filtered_data 
                           if search_term in item['candidate_name'].lower()]
        
        if not filtered_data:
            self.info_text.append("No candidates match the current filters")
            return
        
        # Prepare data for visualization
        positions = np.array([[item['transformer_score'], 
                             item['tfidf_score'], 
                             item['section_score']] for item in filtered_data])
        
        # Set colors based on selected metric
        color_metric = self.color_combo.currentText().lower().replace(' ', '_')
        colors = np.array([item[color_metric] for item in filtered_data])
        colors = (colors - colors.min()) / (colors.max() - colors.min())
        
        # Create color array with alpha channel
        color_array = np.zeros((len(colors), 4))
        color_array[:, 0] = colors  # Red channel
        color_array[:, 1] = 1 - colors  # Green channel
        color_array[:, 2] = 0.5  # Blue channel
        color_array[:, 3] = 1.0  # Alpha channel
        
        # Update scatter plot with larger points and custom colors
        self.scatter.setData(pos=positions, color=color_array, size=15)
        
        # Update axis labels using text items
        self.view.removeItem(self.x_label) if hasattr(self, 'x_label') else None
        self.view.removeItem(self.y_label) if hasattr(self, 'y_label') else None
        self.view.removeItem(self.z_label) if hasattr(self, 'z_label') else None
        
        # Position labels at the edges of the grid
        self.x_label = gl.GLTextItem(pos=(100, 0, 0), text='Transformer Score (%)')
        self.y_label = gl.GLTextItem(pos=(0, 100, 0), text='TFIDF Score (%)')
        self.z_label = gl.GLTextItem(pos=(0, 0, 100), text='Section Score (%)')
        
        self.view.addItem(self.x_label)
        self.view.addItem(self.y_label)
        self.view.addItem(self.z_label)
        
        # Update debug info
        self.info_text.clear()
        self.info_text.append(f"Displaying {len(filtered_data)} candidates")
        self.info_text.append(f"Score threshold: {score_threshold}%")
        self.info_text.append(f"Color metric: {color_metric}")
        if search_term:
            self.info_text.append(f"Search term: {search_term}")
        
        # Add some sample data points
        if len(filtered_data) > 0:
            self.info_text.append("\nSample data points:")
            for item in filtered_data[:5]:  # Show first 5 points
                self.info_text.append(
                    f"\nName: {item['candidate_name']}\n"
                    f"Transformer: {item['transformer_score']:.1f}%\n"
                    f"TFIDF: {item['tfidf_score']:.1f}%\n"
                    f"Section: {item['section_score']:.1f}%\n"
                    f"Combined: {item['combined_score']:.1f}%"
                )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ResumeVisualizationApp()
    window.show()
    sys.exit(app.exec_()) 