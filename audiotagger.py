#!/usr/bin/env python3
"""
Advanced Audio Tagger with AcoustID fingerprinting, smart conflict resolution, and GUI review.
Supports MP3, FLAC, and M4A formats with interactive conflict resolution.
Requires: pip install acoustid musicbrainzngs mutagen tqdm fuzzywuzzy python-Levenshtein PySide6
"""

import os
import sys
import acoustid
import musicbrainzngs
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TRCK, APIC
from mutagen.id3 import ID3NoHeaderError
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from fuzzywuzzy import fuzz
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import argparse
import pickle
from pathlib import Path

# GUI imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
                               QLabel, QGroupBox, QSplitter, QHeaderView, QMessageBox,
                               QProgressBar, QTextEdit, QFileDialog)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor

# Configure MusicBrainz
musicbrainzngs.set_useragent("AdvancedAudioTagger", "2.0", "your@email.com")

# Get your own API key from https://acoustid.org/new-application
ACOUSTID_API_KEY = "YOUR_API_KEY"

class AudioFile:
    """Abstract wrapper for different audio formats."""
    
    @staticmethod
    def load(filepath: str) -> Union[MP3, FLAC, MP4]:
        """Load audio file based on extension."""
        ext = filepath.lower().split('.')[-1]
        if ext == 'mp3':
            return MP3(filepath, ID3=ID3)
        elif ext == 'flac':
            return FLAC(filepath)
        elif ext in ['m4a', 'mp4', 'aac']:
            return MP4(filepath)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    @staticmethod
    def read_tags(filepath: str) -> Dict:
        """Read tags from any supported format."""
        tags = {
            'title': None, 'artist': None, 'album': None, 
            'year': None, 'track': None
        }
        
        try:
            audio = AudioFile.load(filepath)
            ext = filepath.lower().split('.')[-1]
            
            if ext == 'mp3':
                if audio.tags:
                    if 'TIT2' in audio.tags:
                        tags['title'] = str(audio.tags['TIT2'].text[0])
                    if 'TPE1' in audio.tags:
                        tags['artist'] = str(audio.tags['TPE1'].text[0])
                    if 'TALB' in audio.tags:
                        tags['album'] = str(audio.tags['TALB'].text[0])
                    if 'TDRC' in audio.tags:
                        tags['year'] = str(audio.tags['TDRC'].text[0])[:4]
                    if 'TRCK' in audio.tags:
                        track = str(audio.tags['TRCK'].text[0])
                        tags['track'] = track.split('/')[0]
                        
            elif ext == 'flac':
                if 'title' in audio:
                    tags['title'] = audio['title'][0]
                if 'artist' in audio:
                    tags['artist'] = audio['artist'][0]
                if 'album' in audio:
                    tags['album'] = audio['album'][0]
                if 'date' in audio:
                    tags['year'] = audio['date'][0][:4]
                if 'tracknumber' in audio:
                    tags['track'] = audio['tracknumber'][0].split('/')[0]
                    
            elif ext in ['m4a', 'mp4', 'aac']:
                if '\xa9nam' in audio:
                    tags['title'] = audio['\xa9nam'][0]
                if '\xa9ART' in audio:
                    tags['artist'] = audio['\xa9ART'][0]
                if '\xa9alb' in audio:
                    tags['album'] = audio['\xa9alb'][0]
                if '\xa9day' in audio:
                    tags['year'] = audio['\xa9day'][0][:4]
                if 'trkn' in audio:
                    tags['track'] = str(audio['trkn'][0][0])
                    
        except Exception as e:
            logging.debug(f"Error reading tags from {filepath}: {e}")
            
        return tags
    
    @staticmethod
    def write_tags(filepath: str, metadata: Dict, preserve_art: bool = True):
        """Write tags to any supported format."""
        try:
            audio = AudioFile.load(filepath)
            ext = filepath.lower().split('.')[-1]
            
            if ext == 'mp3':
                # Preserve cover art
                existing_art = None
                if preserve_art and audio.tags and 'APIC' in audio.tags:
                    existing_art = audio.tags['APIC']
                
                # Clear and rewrite tags
                audio.delete()
                audio.tags = ID3()
                audio.tags.add(TIT2(encoding=3, text=metadata['title']))
                audio.tags.add(TPE1(encoding=3, text=metadata['artist']))
                if metadata.get('album'):
                    audio.tags.add(TALB(encoding=3, text=metadata['album']))
                if metadata.get('year'):
                    audio.tags.add(TDRC(encoding=3, text=str(metadata['year'])))
                if metadata.get('track'):
                    audio.tags.add(TRCK(encoding=3, text=str(metadata['track'])))
                if existing_art:
                    audio.tags.add(existing_art)
                    
            elif ext == 'flac':
                audio['title'] = metadata['title']
                audio['artist'] = metadata['artist']
                if metadata.get('album'):
                    audio['album'] = metadata['album']
                if metadata.get('year'):
                    audio['date'] = str(metadata['year'])
                if metadata.get('track'):
                    audio['tracknumber'] = str(metadata['track'])
                    
            elif ext in ['m4a', 'mp4', 'aac']:
                audio['\xa9nam'] = metadata['title']
                audio['\xa9ART'] = metadata['artist']
                if metadata.get('album'):
                    audio['\xa9alb'] = metadata['album']
                if metadata.get('year'):
                    audio['\xa9day'] = str(metadata['year'])
                if metadata.get('track'):
                    audio['trkn'] = [(int(metadata['track']), 0)]
                    
            audio.save()
            
        except Exception as e:
            raise Exception(f"Failed to write tags: {e}")


class ConflictReviewGUI(QMainWindow):
    """GUI for reviewing and resolving tagging conflicts."""
    
    decision_made = Signal(str, dict)  # filepath, chosen_metadata
    
    def __init__(self, conflicts: List[Dict]):
        super().__init__()
        self.conflicts = conflicts
        self.current_index = 0
        self.decisions = {}
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Audio Tagging Conflict Review")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Header
        header = QLabel("Review and resolve tagging conflicts")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header)
        
        # Progress
        self.progress_label = QLabel()
        layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Main content area
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)
        
        # Left panel - File info
        left_panel = QGroupBox("File Information")
        left_layout = QVBoxLayout(left_panel)
        
        self.file_label = QLabel()
        self.file_label.setWordWrap(True)
        left_layout.addWidget(self.file_label)
        
        self.existing_tags_text = QTextEdit()
        self.existing_tags_text.setReadOnly(True)
        self.existing_tags_text.setMaximumHeight(150)
        left_layout.addWidget(QLabel("Existing Tags:"))
        left_layout.addWidget(self.existing_tags_text)
        
        left_layout.addStretch()
        splitter.addWidget(left_panel)
        
        # Right panel - Candidates
        right_panel = QGroupBox("Match Candidates")
        right_layout = QVBoxLayout(right_panel)
        
        self.candidates_table = QTableWidget()
        self.candidates_table.setColumnCount(8)
        self.candidates_table.setHorizontalHeaderLabels([
            "Select", "Score", "Artist", "Title", "Album", "Year", "Track", "Source"
        ])
        self.candidates_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.candidates_table.itemClicked.connect(self.on_item_clicked)
        right_layout.addWidget(self.candidates_table)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        # Bottom controls
        controls = QHBoxLayout()
        
        self.skip_btn = QPushButton("Skip File")
        self.skip_btn.clicked.connect(self.skip_file)
        controls.addWidget(self.skip_btn)
        
        self.keep_existing_btn = QPushButton("Keep Existing Tags")
        self.keep_existing_btn.clicked.connect(self.keep_existing)
        controls.addWidget(self.keep_existing_btn)
        
        controls.addStretch()
        
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_conflict)
        controls.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_conflict)
        controls.addWidget(self.next_btn)
        
        self.apply_btn = QPushButton("Apply Selected")
        self.apply_btn.clicked.connect(self.apply_selection)
        self.apply_btn.setEnabled(False)
        controls.addWidget(self.apply_btn)
        
        self.finish_btn = QPushButton("Finish Review")
        self.finish_btn.clicked.connect(self.finish_review)
        controls.addWidget(self.finish_btn)
        
        layout.addLayout(controls)
        
        # Load first conflict
        self.load_conflict()
        
    def load_conflict(self):
        """Load current conflict for review."""
        if self.current_index >= len(self.conflicts):
            return
            
        conflict = self.conflicts[self.current_index]
        
        # Update progress
        self.progress_label.setText(
            f"File {self.current_index + 1} of {len(self.conflicts)}"
        )
        self.progress_bar.setValue(int((self.current_index + 1) / len(self.conflicts) * 100))
        
        # Update file info
        self.file_label.setText(f"File: {conflict['file']}")
        
        # Show existing tags
        existing = conflict['existing_tags']
        existing_text = "\n".join([f"{k}: {v}" for k, v in existing.items() if v])
        self.existing_tags_text.setPlainText(existing_text or "No existing tags")
        
        # Populate candidates table
        self.candidates_table.setRowCount(len(conflict['candidates']))
        
        for i, candidate in enumerate(conflict['candidates']):
            # Radio button for selection
            select_btn = QTableWidgetItem()
            select_btn.setCheckState(Qt.Unchecked)
            self.candidates_table.setItem(i, 0, select_btn)
            
            # Score
            score_item = QTableWidgetItem(f"{candidate['score']:.2f}")
            score_item.setBackground(self.get_score_color(candidate['score']))
            self.candidates_table.setItem(i, 1, score_item)
            
            # Metadata fields
            meta = candidate['metadata']
            self.candidates_table.setItem(i, 2, QTableWidgetItem(meta.get('artist', '')))
            self.candidates_table.setItem(i, 3, QTableWidgetItem(meta.get('title', '')))
            self.candidates_table.setItem(i, 4, QTableWidgetItem(meta.get('album', '')))
            self.candidates_table.setItem(i, 5, QTableWidgetItem(str(meta.get('year', ''))))
            self.candidates_table.setItem(i, 6, QTableWidgetItem(str(meta.get('track', ''))))
            
            # Source info
            details = candidate['details']
            source = f"FP: {details['fingerprint_score']:.2f}, Tag: {details['tag_similarity']:.2f}"
            self.candidates_table.setItem(i, 7, QTableWidgetItem(source))
        
        # Update button states
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.conflicts) - 1)
        self.apply_btn.setEnabled(False)
        
        # Check if we already made a decision for this file
        if conflict['file'] in self.decisions:
            # Restore previous selection
            # Implementation depends on how you want to handle this
            pass
    
    def get_score_color(self, score):
        """Get color based on score value."""
        if score >= 0.9:
            return QColor(200, 255, 200)  # Light green
        elif score >= 0.8:
            return QColor(255, 255, 200)  # Light yellow
        elif score >= 0.7:
            return QColor(255, 220, 200)  # Light orange
        else:
            return QColor(255, 200, 200)  # Light red
    
    def on_item_clicked(self, item):
        """Handle table item click."""
        if item.column() == 0:  # Selection column
            # Uncheck all other items
            for row in range(self.candidates_table.rowCount()):
                if row != item.row():
                    self.candidates_table.item(row, 0).setCheckState(Qt.Unchecked)
            
            # Enable apply button if something is selected
            self.apply_btn.setEnabled(item.checkState() == Qt.Checked)
    
    def get_selected_candidate(self):
        """Get currently selected candidate."""
        for row in range(self.candidates_table.rowCount()):
            if self.candidates_table.item(row, 0).checkState() == Qt.Checked:
                return self.conflicts[self.current_index]['candidates'][row]
        return None
    
    def apply_selection(self):
        """Apply selected candidate."""
        candidate = self.get_selected_candidate()
        if candidate:
            conflict = self.conflicts[self.current_index]
            self.decisions[conflict['file']] = {
                'action': 'apply',
                'metadata': candidate['metadata']
            }
            self.decision_made.emit(conflict['file'], candidate['metadata'])
            
            if self.current_index < len(self.conflicts) - 1:
                self.next_conflict()
            else:
                QMessageBox.information(self, "Complete", "All conflicts reviewed!")
    
    def skip_file(self):
        """Skip current file."""
        conflict = self.conflicts[self.current_index]
        self.decisions[conflict['file']] = {'action': 'skip'}
        
        if self.current_index < len(self.conflicts) - 1:
            self.next_conflict()
    
    def keep_existing(self):
        """Keep existing tags."""
        conflict = self.conflicts[self.current_index]
        self.decisions[conflict['file']] = {'action': 'keep_existing'}
        
        if self.current_index < len(self.conflicts) - 1:
            self.next_conflict()
    
    def prev_conflict(self):
        """Go to previous conflict."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_conflict()
    
    def next_conflict(self):
        """Go to next conflict."""
        if self.current_index < len(self.conflicts) - 1:
            self.current_index += 1
            self.load_conflict()
    
    def finish_review(self):
        """Finish review and close."""
        unreviewed = len(self.conflicts) - len(self.decisions)
        if unreviewed > 0:
            reply = QMessageBox.question(
                self, "Confirm", 
                f"{unreviewed} files not reviewed. Finish anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        self.accept()
        self.close()


class ProcessingThread(QThread):
    """Thread for processing files without blocking GUI."""
    
    progress = Signal(int, str)  # progress_percent, current_file
    conflict_found = Signal(dict)  # conflict_data
    finished = Signal(dict)  # stats
    
    def __init__(self, tagger, files, dry_run=False):
        super().__init__()
        self.tagger = tagger
        self.files = files
        self.dry_run = dry_run
        self.conflicts = []
        
    def run(self):
        """Process files in background."""
        for i, filepath in enumerate(self.files):
            progress_percent = int((i + 1) / len(self.files) * 100)
            self.progress.emit(progress_percent, filepath)
            
            # Process file
            result = self.tagger.process_file(filepath, dry_run=self.dry_run)
            
            # Check for conflicts
            if result and 'conflict' in result:
                self.conflict_found.emit(result['conflict'])
                self.conflicts.append(result['conflict'])
        
        self.finished.emit(self.tagger.stats)


class SmartAudioTagger:
    """Enhanced tagger with multi-format support and GUI integration."""
    
    def __init__(self, api_key, confidence_threshold=0.7, tag_weight=0.3):
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self.tag_weight = tag_weight
        self.stats = {
            'success': 0, 
            'failed': 0, 
            'conflicts_resolved': 0,
            'used_existing_tags': 0,
            'skipped': 0
        }
        self.conflict_log = []
        self.supported_formats = {'.mp3', '.flac', '.m4a', '.mp4', '.aac'}
        
    def read_existing_tags(self, filepath):
        """Read existing tags from any supported format."""
        tags = AudioFile.read_tags(filepath)
        
        # Try filename parsing as fallback
        filename_tags = self._parse_filename(filepath)
        for key, value in filename_tags.items():
            if not tags[key] and value:
                tags[key] = value
                
        return tags
    
    def _parse_filename(self, filepath):
        """Parse artist/title from filename."""
        filename = os.path.splitext(os.path.basename(filepath))[0]
        tags = {'title': None, 'artist': None, 'album': None, 'year': None, 'track': None}
        
        import re
        patterns = [
            r'^(?P<artist>[^-]+)\s*-\s*(?P<title>.+)$',
            r'^(?P<track>\d+)\.?\s*(?P<artist>[^-]+)\s*-\s*(?P<title>.+)$',
            r'^(?P<track>\d+)\s*-\s*(?P<title>.+)$',
            r'^(?P<title>.+)$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                for key, value in match.groupdict().items():
                    if value:
                        tags[key] = value.strip()
                break
                
        return tags
    
    def calculate_tag_similarity(self, existing_tags: Dict, candidate: Dict) -> float:
        """Calculate similarity between existing tags and candidate."""
        scores = []
        weights = {
            'artist': 0.4,
            'title': 0.3,
            'album': 0.2,
            'year': 0.1
        }
        
        for field, weight in weights.items():
            existing = existing_tags.get(field)
            candidate_val = candidate.get(field)
            
            if existing and candidate_val:
                if field == 'year':
                    score = 1.0 if str(existing) == str(candidate_val) else 0.0
                else:
                    score = fuzz.ratio(str(existing).lower(), 
                                     str(candidate_val).lower()) / 100.0
                scores.append(score * weight)
            elif not existing:
                scores.append(0.5 * weight)
                
        return sum(scores) / sum(weights.values())
    
    def score_match(self, candidate: Dict, fingerprint_score: float, 
                   existing_tags: Dict) -> Tuple[float, Dict]:
        """Calculate combined score for a match."""
        fp_score = fingerprint_score
        tag_score = self.calculate_tag_similarity(existing_tags, candidate)
        
        combined_score = (
            fp_score * (1 - self.tag_weight) + 
            tag_score * self.tag_weight
        )
        
        scoring_details = {
            'fingerprint_score': fp_score,
            'tag_similarity': tag_score,
            'combined_score': combined_score,
            'existing_tags_used': any(existing_tags.values())
        }
        
        return combined_score, scoring_details
    
    def find_best_match(self, results, existing_tags):
        """Find best match with conflict detection."""
        candidates = []
        
        for result in results:
            fp_score = result.get('score', 0)
            
            if fp_score < self.confidence_threshold:
                continue
                
            for recording in result.get('recordings', []):
                if all(k in recording for k in ['title', 'artists']):
                    candidate = {
                        'title': recording['title'],
                        'artist': recording['artists'][0]['name'],
                        'album': None,
                        'year': None,
                        'track': None,
                        'musicbrainz_id': recording.get('id')
                    }
                    
                    for release in recording.get('releases', []):
                        if 'title' in release:
                            candidate['album'] = release['title']
                            if 'date' in release:
                                candidate['year'] = release['date'].get('year')
                            if 'mediums' in release:
                                for medium in release['mediums']:
                                    for t in medium.get('tracks', []):
                                        if t.get('recording_id') == recording['id']:
                                            candidate['track'] = t.get('position')
                            break
                    
                    combined_score, scoring_details = self.score_match(
                        candidate, fp_score, existing_tags
                    )
                    
                    candidates.append({
                        'metadata': candidate,
                        'score': combined_score,
                        'details': scoring_details
                    })
        
        if not candidates:
            return None, None, None
            
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Check for conflicts
        conflict_data = None
        if len(candidates) > 1:
            score_diff = candidates[0]['score'] - candidates[1]['score']
            if score_diff < 0.15:  # Close match - conflict!
                conflict_data = {
                    'existing_tags': existing_tags,
                    'candidates': candidates[:5],  # Top 5 candidates
                    'score_difference': score_diff
                }
                self.stats['conflicts_resolved'] += 1
        
        best = candidates[0]
        if best['details']['existing_tags_used']:
            self.stats['used_existing_tags'] += 1
            
        return best['metadata'], best['details'], conflict_data
    
    def process_file(self, filepath, dry_run=False):
        """Process a single audio file."""
        try:
            # Check format
            ext = Path(filepath).suffix.lower()
            if ext not in self.supported_formats:
                logging.warning(f"Unsupported format: {filepath}")
                return None
            
            # Read existing tags
            existing_tags = self.read_existing_tags(filepath)
            
            logging.info(f"Processing: {os.path.basename(filepath)}")
            if any(existing_tags.values()):
                logging.debug(f"  Existing tags: {existing_tags}")
            
            # Generate fingerprint
            duration, fingerprint = acoustid.fingerprint_file(filepath)
            
            # Lookup in AcoustID
            results = acoustid.lookup(self.api_key, fingerprint, duration,
                                    meta='recordings releases')
            
            # Find best match
            best_match, scoring_details, conflict_data = self.find_best_match(
                results, existing_tags
            )
            
            if not best_match:
                if existing_tags.get('artist') and existing_tags.get('title'):
                    logging.info(f"  No match found, keeping existing tags")
                    self.stats['success'] += 1
                    return {'status': 'kept_existing'}
                else:
                    logging.warning(f"  No match found for {filepath}")
                    self.stats['failed'] += 1
                    return {'status': 'failed'}
            
            # Log decision
            if scoring_details:
                logging.debug(f"  Match scores: FP={scoring_details['fingerprint_score']:.2f}, "
                            f"Tag={scoring_details['tag_similarity']:.2f}, "
                            f"Combined={scoring_details['combined_score']:.2f}")
            
            # If conflict detected, prepare data for GUI
            if conflict_data:
                conflict_data['file'] = filepath
                self.conflict_log.append(conflict_data)
                return {'status': 'conflict', 'conflict': conflict_data, 'best_match': best_match}
            
            # Apply tags if not dry run
            if not dry_run:
                AudioFile.write_tags(filepath, best_match)
                logging.info(f"  Tagged: {best_match['artist']} - {best_match['title']}")
            else:
                logging.info(f"  [DRY RUN] Would tag: {best_match['artist']} - {best_match['title']}")
            
            self.stats['success'] += 1
            return {'status': 'success', 'metadata': best_match}
            
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")
            self.stats['failed'] += 1
            return {'status': 'error', 'error': str(e)}
    
    def process_directory(self, directory, max_files=None, dry_run=False, 
                         interactive=False, max_workers=4):
        """Process all audio files in directory."""
        # Collect all audio files
        audio_files = []
        for root, _, files in os.walk(directory):
            for f in files:
                if Path(f).suffix.lower() in self.supported_formats:
                    audio_files.append(os.path.join(root, f))
        
        # Apply file limit if specified
        if max_files:
            audio_files = audio_files[:max_files]
        
        print(f"Found {len(audio_files)} audio files")
        if dry_run:
            print("DRY RUN MODE - No files will be modified")
        print(f"Tag weight: {self.tag_weight} (0=fingerprint only, 1=tags only)")
        
        # Process files with progress bar
        conflicts = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i, filepath in enumerate(audio_files):
                # Rate limit
                if i > 0 and i % 3 == 0:
                    time.sleep(1)
                
                future = executor.submit(self.process_file, filepath, dry_run)
                futures.append((future, filepath))
            
            # Progress bar
            for future, filepath in tqdm(futures, desc="Processing files"):
                try:
                    result = future.result(timeout=30)
                    if result and result.get('status') == 'conflict':
                        conflicts.append(result['conflict'])
                except Exception as e:
                    logging.error(f"Timeout/error for {filepath}: {e}")
        
        # Handle conflicts interactively if requested
        if interactive and conflicts:
            print(f"\nFound {len(conflicts)} conflicts to review")
            app = QApplication.instance() or QApplication(sys.argv)
            
            gui = ConflictReviewGUI(conflicts)
            
            # Connect signal to handle decisions
            def apply_decision(filepath, metadata):
                if not dry_run:
                    AudioFile.write_tags(filepath, metadata)
                    logging.info(f"Applied user decision for {filepath}")
            
            gui.decision_made.connect(apply_decision)
            gui.show()
            app.exec()
            
            # Update stats based on decisions
            for filepath, decision in gui.decisions.items():
                if decision['action'] == 'skip':
                    self.stats['skipped'] += 1
                elif decision['action'] in ['apply', 'keep_existing']:
                    self.stats['success'] += 1
        
        # Print final stats
        self.print_stats(len(audio_files))
        
        # Save reports
        self.save_reports(dry_run)
    
    def print_stats(self, total_files):
        """Print processing statistics."""
        print(f"\nResults:")
        print(f"  Successfully tagged: {self.stats['success']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Skipped: {self.stats['skipped']}")
        print(f"  Conflicts found: {len(self.conflict_log)}")
        print(f"  Used existing tags: {self.stats['used_existing_tags']}")
        if total_files > 0:
            print(f"  Success rate: {self.stats['success'] / total_files * 100:.1f}%")
    
    def save_reports(self, dry_run=False):
        """Save processing reports."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save conflict log
        if self.conflict_log:
            log_file = f"tag_conflicts_{timestamp}.json"
            with open(log_file, 'w') as f:
                json.dump(self.conflict_log, f, indent=2)
            print(f"\nConflict log saved to: {log_file}")
        
        # Save detailed report
        report_file = f"tagging_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("Audio Tagging Report\n")
            f.write("===================\n\n")
            if dry_run:
                f.write("DRY RUN - No files were modified\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total files: {sum(self.stats.values())}\n")
            for key, value in self.stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Smart audio tagger with AcoustID fingerprinting"
    )
    parser.add_argument("directory", help="Directory containing audio files")
    parser.add_argument("--api-key", default=ACOUSTID_API_KEY, 
                       help="AcoustID API key")
    parser.add_argument("--confidence", type=float, default=0.7,
                       help="Minimum fingerprint confidence (0-1)")
    parser.add_argument("--tag-weight", type=float, default=0.3,
                       help="Weight for existing tags (0-1)")
    parser.add_argument("--max-files", type=int, 
                       help="Maximum number of files to process")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview changes without modifying files")
    parser.add_argument("--interactive", action="store_true",
                       help="Enable GUI for conflict resolution")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'audio_tagger_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create tagger
    tagger = SmartAudioTagger(
        args.api_key, 
        args.confidence, 
        args.tag_weight
    )
    
    # Process directory
    tagger.process_directory(
        args.directory,
        max_files=args.max_files,
        dry_run=args.dry_run,
        interactive=args.interactive,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
