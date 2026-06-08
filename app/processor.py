from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import re
import uuid
import hashlib

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        # Using tiktoken for accurate token-based splitting
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-3.5-turbo",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def infer_severity(self, text):
        """Infer severity based on keywords"""
        text_lower = text.lower()
        high_keywords = ["danger", "critical", "risk", "failure", "fatal", "severe"]
        medium_keywords = ["warning", "caution", "attention", "important"]
        
        if any(k in text_lower for k in high_keywords):
            return "high"
        if any(k in text_lower for k in medium_keywords):
            return "medium"
        return "low"

    def infer_escalation(self, text, severity):
        """Infer if escalation is required"""
        if severity == "high":
            return True
        
        text_lower = text.lower()
        escalation_phrases = [
            "contact service center",
            "authorized technician required",
            "do not attempt repair",
            "see your dealer",
            "professional assistance needed"
        ]
        return any(p in text_lower for p in escalation_phrases)

    def extract_and_chunk(self, file_path):
        """Extract text and perform semantic + structural chunking"""
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            full_content = "\n".join([doc.page_content for doc in documents])
        elif file_extension in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # 1. Structural Segmentation
        # Split by headings: ALL CAPS, Title Case with numbers, or special keywords
        section_pattern = r'^(?:(?:[0-9.]+\s+)?[A-Z][A-Z\s]{3,}|(?:[0-9.]+\s+)[A-Z][a-z].*|WARNING:|CAUTION:|NOTE:|PROCEDURE:|TROUBLESHOOTING:)'
        
        segments = re.split(f'({section_pattern})', full_content, flags=re.MULTILINE)
        
        all_chunks = []
        current_section = "General"
        
        # segments will be [pre-text, header1, content1, header2, content2, ...]
        if segments[0].strip():
            self._process_segment(segments[0], current_section, file_name, all_chunks)

        for i in range(1, len(segments), 2):
            header = segments[i].strip()
            content = segments[i+1] if i+1 < len(segments) else ""
            
            # Update section name if it looks like a major header (not a warning)
            if not any(k in header.upper() for k in ["WARNING", "CAUTION", "NOTE", "PROCEDURE"]):
                current_section = header
            
            combined_text = f"{header}\n{content}"
            self._process_segment(combined_text, current_section, file_name, all_chunks)

        return all_chunks

    def _process_segment(self, text, section, file_name, all_chunks):
        """Sub-chunking for a semantic segment"""
        if not text.strip():
            return

        # Use recursive character splitter (token-aware)
        sub_chunks = self.text_splitter.split_text(text)
        
        for idx, chunk_text in enumerate(sub_chunks):
            severity = self.infer_severity(chunk_text)
            escalation = self.infer_escalation(chunk_text, severity)
            
            # Deterministic issue_id and chunk_id
            file_slug = re.sub(r'[^a-z0-9]', '-', file_name.lower())[:20]
            unique_seed = f"{section}-{len(all_chunks)}-{file_name}"
            issue_id = f"ISSUE-{hashlib.md5(unique_seed.encode()).hexdigest()[:8].upper()}"
            
            all_chunks.append({
                "chunk_id": f"{file_slug}_{hashlib.md5(chunk_text.encode()).hexdigest()[:10]}",
                "issue_id": issue_id,
                "text": chunk_text.strip(),
                "section": section,
                "severity": severity,
                "escalation_required": escalation,
                "chunk_index": len(all_chunks)
            })
