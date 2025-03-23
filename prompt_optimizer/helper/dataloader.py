from typing import List, Dict, Any, Iterator, Optional
import json
import os
import pandas as pd
import logging
from pathlib import Path
import random
class DataLoader:
    """
    A class that loads evaluation data (input, output, system_prompt) and provides chunk-based access.
    """
    
    def __init__(self, 
                 data_path: str = None, 
                 data: List[Dict[str, Any]] = None,
                 shuffle: bool = True,
                 seed: int = 42):
        """
        Initialize the DataLoader with either a path to data file or direct data.
        
        Args:
            data_path: Path to the data file (JSON)
            data: Direct data input as a list of dictionaries
        """
        if data_path is not None:
            self.data = self._load_data_from_csv(data_path, shuffle, seed)
        elif data is not None:
            self.data = data
        else:
            raise ValueError("Either data_path or data must be provided")
            
        self._validate_data()
        self.current_index = 0
        
    def _load_data_from_csv(self, 
                            data_path: str, 
                            shuffle: bool, 
                            seed: int) -> List[Dict[str, Any]]:
        """
        Load data from a file.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            List of data items
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        data = pd.read_csv(path)
        data = data.to_dict(orient='records')
        if shuffle:
            random.seed(seed)
            random.shuffle(data)
        return data
    
    def _validate_data(self):
        """Validate that data contains required fields."""
        required_fields = ['input', 'ground_truth']
        
        for i, item in enumerate(self.data):
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                logging.warning(f"Item {i} is missing required fields: {missing_fields}")
    
    def get_chunk(self, chunk_size: int) -> List[Dict[str, Any]]:
        """
        Get a chunk of data of specified size.
        
        Args:
            chunk_size: Number of data items to return
            
        Returns:
            List of data items
        """
        if self.current_index >= len(self.data):
            return []
            
        chunk = self.data[self.current_index:self.current_index + chunk_size]
        self.current_index += chunk_size
        return chunk
    
    def get_chunks(self, chunk_size: int) -> Iterator[List[Dict[str, Any]]]:
        """
        Get an iterator that yields chunks of data.
        
        Args:
            chunk_size: Size of each chunk
            
        Yields:
            Chunks of data
        """
        self.current_index = 0
        while True:
            chunk = self.get_chunk(chunk_size)
            if not chunk:
                break
            yield chunk
            
    def reset(self):
        """Reset the current index to start from the beginning."""
        self.current_index = 0
        
    def __len__(self):
        """Return the total number of data items."""
        return len(self.data)
    
if __name__ == "__main__":
    sample_data = DataLoader(data_path='sample_data.csv')
    print(sample_data.get_chunk(2))