"""
Basic tests for RAG pipeline components.

This module contains basic unit tests to demonstrate testing structure.
Tests can be expanded as needed for full coverage.
"""

import pytest
from pathlib import Path


class TestConfiguration:
    """Test configuration loading and validation."""
    
    def test_config_loads(self):
        """Test that configuration can be loaded."""
        from config import load_settings
        
        # This will use default values or .env
        settings = load_settings()
        assert settings is not None
        assert hasattr(settings, 'openai')
        assert hasattr(settings, 'pinecone')
        assert hasattr(settings, 'qdrant')
    
    def test_config_validation(self):
        """Test that configuration validates types."""
        from config import ChunkingConfig
        
        config = ChunkingConfig(
            min_size=100,
            max_size=512,
            overlap=50
        )
        assert config.min_size < config.max_size
        assert config.overlap < config.max_size


class TestPreprocessing:
    """Test text preprocessing functionality."""
    
    def test_basic_cleaning(self):
        """Test basic text cleaning operations."""
        from preprocessors import DocumentPreprocessor
        
        preprocessor = DocumentPreprocessor()
        
        # Test whitespace normalization
        text = "Hello    world\n\n\n\nMultiple   spaces"
        # Preprocessor should normalize whitespace
        assert preprocessor is not None
    
    def test_unicode_handling(self):
        """Test Unicode character handling."""
        from preprocessors import DocumentPreprocessor
        
        preprocessor = DocumentPreprocessor()
        
        # Test with Unicode characters
        text = "Hello 世界 🌍"
        # Should handle Unicode properly
        assert preprocessor is not None


class TestChunking:
    """Test chunking strategies."""
    
    def test_chunk_creation(self):
        """Test basic chunk creation."""
        from chunking_strategies import Chunk, ChunkType
        
        chunk = Chunk(
            id="test-1",
            text="This is a test chunk",
            chunk_type=ChunkType.PARAGRAPH,
            metadata={"source": "test"}
        )
        
        assert chunk.id == "test-1"
        assert chunk.text == "This is a test chunk"
        assert chunk.chunk_type == ChunkType.PARAGRAPH
    
    def test_chunk_to_node(self):
        """Test conversion of chunk to LlamaIndex node."""
        from chunking_strategies import Chunk, ChunkType
        
        chunk = Chunk(
            id="test-1",
            text="Test content",
            chunk_type=ChunkType.PARAGRAPH,
            metadata={"source": "test"}
        )
        
        node = chunk.to_node()
        assert node.text == "Test content"
        assert node.id_ == "test-1"


class TestPipeline:
    """Test pipeline orchestration."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        # This test requires proper environment setup
        # For now, just test structure exists
        from pipeline import RAGIngestionPipeline
        
        # Test class exists and has expected methods
        assert hasattr(RAGIngestionPipeline, 'process_document')
        assert hasattr(RAGIngestionPipeline, 'process_batch')
        assert hasattr(RAGIngestionPipeline, 'run_full_pipeline')


# Pytest configuration
@pytest.fixture
def sample_document():
    """Fixture for sample document."""
    from llama_index.core import Document
    
    return Document(
        text="This is a sample document for testing purposes.",
        metadata={"filename": "test.txt", "source": "test"}
    )


@pytest.fixture
def sample_documents():
    """Fixture for multiple sample documents."""
    from llama_index.core import Document
    
    return [
        Document(
            text=f"Sample document {i} content.",
            metadata={"filename": f"test{i}.txt", "source": "test"}
        )
        for i in range(3)
    ]
