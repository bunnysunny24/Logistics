from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from loguru import logger

class LocalHuggingFaceLLM(LLM):
    """LLM wrapper for local HuggingFace models."""
    
    model_name: str = "facebook/opt-125m"  # Default to a smaller model
    tokenizer_name: Optional[str] = None
    device: Optional[str] = None
    model_kwargs: Dict[str, Any] = {}
    
    def __init__(self, model_name=None, **kwargs):
        """Initialize the local LLM."""
        super().__init__(**kwargs)
        
        # Initialize instance variables
        self._model = None
        self._tokenizer = None
        self.pipeline = None
        
        if model_name:
            self.model_name = model_name
            
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name
            
        # Determine device (CPU/GPU)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Initializing local LLM with model: {self.model_name} on {self.device}")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        try:
            # Load the model and tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            
            # Load with lower precision for memory efficiency
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True if self.device == "cpu" else False,
                **self.model_kwargs
            }
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move model to the appropriate device
            self._model.to(self.device)
            logger.info(f"Successfully loaded model {self.model_name}")
            
            # Create the text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Successfully initialized text generation pipeline")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            # Set pipeline to None so we know it failed
            self.pipeline = None
            raise e
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text based on prompt."""
        try:
            # Get generation parameters with defaults
            max_new_tokens = kwargs.get("max_new_tokens", 512)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.9)
            
            # Generate text
            generations = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                num_return_sequences=1,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            
            # Extract generated text
            text = generations[0]["generated_text"]
            
            # Remove the prompt from the beginning of the text
            if text.startswith(prompt):
                text = text[len(prompt):]
                
            # Apply stopping criteria if provided
            if stop:
                for stop_seq in stop:
                    if stop_seq in text:
                        text = text[:text.find(stop_seq)]
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "local_huggingface"
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "device": self.device
        }