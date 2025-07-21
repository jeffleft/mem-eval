import re
import os
from typing import List, Dict, Any, Optional


class GrepTool:
    """Text pattern search tool using regex"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []
    
    def load_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Load documents for searching"""
        self.documents = documents
        self.metadata = metadata or [{"id": i} for i in range(len(documents))]
    
    def search(self, pattern: str, case_sensitive: bool = False, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for pattern in documents using regex"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        flags = 0 if case_sensitive else re.IGNORECASE
        
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        
        results = []
        for i, doc in enumerate(self.documents):
            matches = list(regex.finditer(doc))
            if matches:
                # Get context around matches
                match_details = []
                for match in matches[:5]:  # Limit matches per document
                    start = max(0, match.start() - 50)
                    end = min(len(doc), match.end() + 50)
                    context = doc[start:end]
                    
                    match_details.append({
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "context": context
                    })
                
                results.append({
                    "document_id": i,
                    "document": doc,
                    "metadata": self.metadata[i],
                    "match_count": len(matches),
                    "matches": match_details
                })
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def search_exact(self, text: str, case_sensitive: bool = False, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for exact text matches"""
        if not case_sensitive:
            text = text.lower()
            documents = [doc.lower() for doc in self.documents]
        else:
            documents = self.documents
        
        results = []
        for i, doc in enumerate(documents):
            if text in doc:
                # Find all occurrences
                start = 0
                occurrences = []
                while True:
                    pos = doc.find(text, start)
                    if pos == -1:
                        break
                    
                    # Get context
                    context_start = max(0, pos - 50)
                    context_end = min(len(doc), pos + len(text) + 50)
                    context = self.documents[i][context_start:context_end]  # Use original case
                    
                    occurrences.append({
                        "position": pos,
                        "context": context
                    })
                    
                    start = pos + 1
                    if len(occurrences) >= 5:  # Limit occurrences per document
                        break
                
                results.append({
                    "document_id": i,
                    "document": self.documents[i],
                    "metadata": self.metadata[i],
                    "occurrence_count": len(occurrences),
                    "occurrences": occurrences
                })
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def search_keywords(self, keywords: List[str], case_sensitive: bool = False, 
                       match_all: bool = False, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for multiple keywords"""
        if not case_sensitive:
            keywords = [kw.lower() for kw in keywords]
            documents = [doc.lower() for doc in self.documents]
        else:
            documents = self.documents
        
        results = []
        for i, doc in enumerate(documents):
            found_keywords = []
            for keyword in keywords:
                if keyword in doc:
                    found_keywords.append(keyword)
            
            # Check matching criteria
            if match_all and len(found_keywords) == len(keywords):
                # All keywords found
                pass
            elif not match_all and found_keywords:
                # At least one keyword found
                pass
            else:
                continue
            
            results.append({
                "document_id": i,
                "document": self.documents[i],
                "metadata": self.metadata[i],
                "found_keywords": found_keywords,
                "keyword_count": len(found_keywords)
            })
            
            if len(results) >= max_results:
                break
        
        return results