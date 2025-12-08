import os
import io
import time
import requests
import bittensor as bt
from typing import Dict, Any, Optional

from PIL import Image
import c2pa

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class Models:
    """Symbolic constants for Leonardo.ai models."""

    PHOENIX = "leonardo-phoenix"
    DIFFUSION_XL = "leonardo-diffusion-xl"
    VISION = "leonardo-vision"
    KINO = "leonardo-kino"


MODEL_INFO = {
    Models.PHOENIX: {
        "endpoint": "https://cloud.leonardo.ai/api/rest/v1/generations",
        "name": "Leonardo Phoenix",
        "family": "phoenix",
        "description": "High-quality image generation model",
    },
    Models.DIFFUSION_XL: {
        "endpoint": "https://cloud.leonardo.ai/api/rest/v1/generations",
        "name": "Leonardo Diffusion XL",
        "family": "diffusion-xl",
        "description": "XL-scale diffusion model",
    },
    Models.VISION: {
        "endpoint": "https://cloud.leonardo.ai/api/rest/v1/generations",
        "name": "Leonardo Vision",
        "family": "vision",
        "description": "Vision-focused model",
    },
    Models.KINO: {
        "endpoint": "https://cloud.leonardo.ai/api/rest/v1/generations",
        "name": "Leonardo Kino",
        "family": "kino",
        "description": "Kino model variant",
    },
}


class LeonardoService(BaseGenerationService):
    """
    Leonardo.ai generation service for image generation.

    Features:
    - Asynchronous image generation with polling
    - Format detection and conversion (JPG to PNG)
    - C2PA metadata extraction (note: Leonardo.ai does not embed C2PA)
    - Model → endpoint mapping
    """

    # Allowed output formats
    VALID_FORMATS = {"png", "jpeg", "webp"}

    def __init__(self, config: Any = None):
        super().__init__(config)

        self.api_key = os.getenv("LEONARDO_API_KEY")
        self.timeout = 90
        self.default_model = Models.PHOENIX

        if not self.api_key:
            bt.logging.warning("LEONARDO_API_KEY not found.")
        else:
            bt.logging.info("LeonardoService initialized with API key")

    # ---------------------------------------------------------------------
    # Base methods
    # ---------------------------------------------------------------------
    def is_available(self) -> bool:
        return self.api_key is not None and self.api_key.strip() != ""

    def supports_modality(self, modality: str) -> bool:
        return modality == "image"

    def get_supported_tasks(self) -> Dict[str, list]:
        return {
            "image": ["image_generation"],
            "video": [],  # Leonardo.ai doesn't support video generation yet
        }

    def get_api_key_requirements(self) -> Dict[str, str]:
        return {"LEONARDO_API_KEY": "API key for Leonardo.ai image generation"}

    # ---------------------------------------------------------------------
    # Processing logic
    # ---------------------------------------------------------------------
    def process(self, task: GenerationTask) -> Dict[str, Any]:
        if task.modality != "image":
            raise ValueError(f"LeonardoService does not support modality: {task.modality}")

        return self._generate_image(task)

    # ---------------------------------------------------------------------
    # C2PA extractor
    # ---------------------------------------------------------------------
    def _extract_c2pa_metadata(self, img_bytes: bytes, output_format: str) -> Optional[Dict[str, Any]]:
        """Extract embedded C2PA manifest from image bytes."""

        mime_map = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "webp": "image/webp",
        }

        mime_type = mime_map.get(output_format.lower(), "application/octet-stream")

        try:
            with io.BytesIO(img_bytes) as f:
                with c2pa.Reader(mime_type, f) as reader:
                    return reader.json()
        except Exception as e:
            bt.logging.warning(f"No C2PA metadata detected or failed to read: {e}")
            return None

    def _get_endpoint(self, model: str) -> str:
        if model not in MODEL_INFO:
            raise ValueError(f"Unknown Leonardo.ai model: {model}")
        return MODEL_INFO[model]["endpoint"]

    # ---------------------------------------------------------------------
    # Image generation core
    # ---------------------------------------------------------------------
    def _generate_image(self, task: GenerationTask) -> Dict[str, Any]:
        try:
            params = task.parameters or {}

            model = params.get("model", self.default_model)
            prompt = task.prompt

            bt.logging.info(f"Leonardo.ai generating image with model={model}")

            if model not in MODEL_INFO:
                raise ValueError(
                    f"Unknown Leonardo.ai model: {model}. "
                    f"Available models: {list(MODEL_INFO.keys())}"
                )

            url = self._get_endpoint(model)

            output_format = params.get("format", "png")
            if output_format not in self.VALID_FORMATS:
                output_format = "png"

            # Leonardo.ai API payload
            api_data = {
                "prompt": prompt,
                "num_images": 1,
            }

            # Optional parameters
            if "negative_prompt" in params:
                api_data["negative_prompt"] = params["negative_prompt"]
            if "seed" in params:
                api_data["seed"] = params["seed"]
            if "width" in params:
                api_data["width"] = params["width"]
            if "height" in params:
                api_data["height"] = params["height"]
            
            # Convert aspect_ratio to width/height if provided
            if "aspect_ratio" in params:
                aspect_ratio = params["aspect_ratio"]
                # Common aspect ratios to width/height mapping
                aspect_map = {
                    "16:9": (1024, 576),
                    "9:16": (576, 1024),
                    "1:1": (1024, 1024),
                    "4:3": (1024, 768),
                    "3:4": (768, 1024),
                }
                if aspect_ratio in aspect_map:
                    width, height = aspect_map[aspect_ratio]
                    api_data["width"] = width
                    api_data["height"] = height
                else:
                    bt.logging.warning(f"Unknown aspect_ratio {aspect_ratio}, using default dimensions")

            # Leonardo.ai uses Bearer token authentication
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "accept": "application/json",
            }

            start_time = time.time()
            response = requests.post(
                url,
                headers=headers,
                json=api_data,
                timeout=self.timeout,
            )

            if response.status_code == 401:
                # Try alternative auth format
                bt.logging.warning("Bearer token auth failed, trying alternative format...")
                headers_alt = {
                    "Authorization": self.api_key,
                    "Content-Type": "application/json",
                    "accept": "application/json",
                }
                response = requests.post(
                    url,
                    headers=headers_alt,
                    json=api_data,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    headers = headers_alt
                else:
                    raise RuntimeError(
                        f"Leonardo.ai API authentication failed {response.status_code}: {response.text}\n"
                        f"Please check:\n"
                        f"1. Your API key is correct (first 10 chars: {self.api_key[:10] if self.api_key else 'None'}...)\n"
                        f"2. API key is set as LEONARDO_API_KEY environment variable\n"
                        f"3. Get API key from: https://app.leonardo.ai/settings"
                    )
            
            if response.status_code != 200:
                error_msg = f"Leonardo.ai API error {response.status_code}: {response.text}"
                if response.status_code == 500 and "authorization" in response.text.lower():
                    error_msg += (
                        f"\n\n⚠️  Authentication error detected. Please verify:\n"
                        f"1. Your LEONARDO_API_KEY is valid and active\n"
                        f"2. API key format is correct (should start with your account identifier)\n"
                        f"3. Get your API key from: https://app.leonardo.ai/settings\n"
                        f"4. API key (first 10 chars): {self.api_key[:10] if self.api_key and len(self.api_key) > 10 else 'Too short or missing'}...\n"
                        f"5. Check Leonardo.ai API documentation: https://docs.leonardo.ai/"
                    )
                raise RuntimeError(error_msg)

            response_data = response.json()
            # Leonardo.ai returns generationId nested in sdGenerationJob
            generation_id = None
            if "sdGenerationJob" in response_data:
                generation_id = response_data["sdGenerationJob"].get("generationId")
            elif "generationId" in response_data:
                generation_id = response_data["generationId"]
            elif "generation_id" in response_data:
                generation_id = response_data["generation_id"]
            
            if not generation_id:
                # Check if image is directly in response
                if "images" in response_data and len(response_data["images"]) > 0:
                    image_url = response_data["images"][0].get("url") or response_data["images"][0]
                    detected_format = self._detect_format_from_url(image_url)
                    img_bytes = self._download_image(image_url, headers)
                    # Convert if needed
                    if detected_format != output_format.lower():
                        bt.logging.info(f"Converting image from {detected_format} to {output_format}")
                        img_bytes = self._convert_image_format(img_bytes, detected_format, output_format)
                        detected_format = output_format.lower()
                    # Extract C2PA with correct format
                    c2pa_metadata = self._extract_c2pa_metadata(img_bytes, detected_format)
                    gen_time = time.time() - start_time
                    return {
                        "data": img_bytes,
                        "metadata": {
                            "model": model,
                            "provider": "leonardo.ai",
                            "format": detected_format.upper(),
                            "generation_time": gen_time,
                            "c2pa": c2pa_metadata,
                        },
                    }
                else:
                    raise RuntimeError(f"Leonardo.ai API did not return generationId or images: {response_data}")
            else:
                # Poll for completion
                img_bytes, actual_format = self._poll_for_result(generation_id, headers)
                
                # Convert to requested format if different from actual format
                if actual_format != output_format.lower():
                    bt.logging.info(f"Converting image from {actual_format} to {output_format}")
                    img_bytes = self._convert_image_format(img_bytes, actual_format, output_format)
                    actual_format = output_format.lower()

                gen_time = time.time() - start_time
                # Extract C2PA (embedded in image) - use actual format
                c2pa_metadata = self._extract_c2pa_metadata(img_bytes, actual_format)

                # Return final miner-compatible result
                return {
                    "data": img_bytes,
                    "metadata": {
                        "model": model,
                        "provider": "leonardo.ai",
                        "format": actual_format.upper(),
                        "generation_time": gen_time,
                        "c2pa": c2pa_metadata,
                    },
                }

        except Exception as e:
            bt.logging.error(f"Leonardo.ai image generation failed: {e}")
            raise

    def _detect_format_from_url(self, url: str) -> str:
        """Detect image format from URL extension."""
        url_lower = url.lower()
        if url_lower.endswith(".jpg") or url_lower.endswith(".jpeg"):
            return "jpeg"
        elif url_lower.endswith(".png"):
            return "png"
        elif url_lower.endswith(".webp"):
            return "webp"
        else:
            # Default to jpeg for Leonardo.ai (they typically return JPG)
            return "jpeg"
    
    def _convert_image_format(self, img_bytes: bytes, from_format: str, to_format: str) -> bytes:
        """Convert image from one format to another."""
        try:
            img = Image.open(io.BytesIO(img_bytes))
            # Convert RGBA to RGB if converting to JPEG
            if to_format.lower() == "jpeg" and img.mode == "RGBA":
                # Create white background
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
                img = rgb_img
            
            output = io.BytesIO()
            img.save(output, format=to_format.upper())
            return output.getvalue()
        except Exception as e:
            bt.logging.warning(f"Failed to convert image from {from_format} to {to_format}: {e}")
            return img_bytes  # Return original if conversion fails
    
    def _poll_for_result(self, generation_id: str, headers: Dict[str, str]) -> tuple[bytes, str]:
        """Poll for generation completion and return image bytes with detected format."""
        status_url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}"
        bt.logging.info(f"Polling for generation completion: {generation_id}")

        max_attempts = 60
        poll_interval = 2

        for attempt in range(max_attempts):
            try:
                bt.logging.info(f"Polling for generation completion: {generation_id} at Attemp #{attempt}")
                response = requests.get(status_url, headers=headers, timeout=30)
                if response.status_code != 200:
                    bt.logging.warning(
                        f"Status check failed: {response.status_code} - {response.text}"
                    )
                    time.sleep(poll_interval)
                    continue

                try:
                    status_data = response.json()
                except ValueError as e:
                    bt.logging.error(f"Failed to parse JSON response: {e}")
                    time.sleep(poll_interval)
                    continue

                # Check status - Leonardo.ai uses generations_by_pk structure
                generation_data = status_data.get("generations_by_pk", {})
                if not generation_data:
                    # Fallback to direct status field
                    generation_data = status_data
                
                status = generation_data.get("status", "").upper()
                status_lower = status.lower()

                if status_lower in ("completed", "success", "done", "succeeded", "complete"):
                    # Get image URL from generated_images array
                    generated_images = generation_data.get("generated_images", [])
                    
                    if generated_images and len(generated_images) > 0:
                        image_data = generated_images[0]
                        if isinstance(image_data, dict):
                            image_url = (
                                image_data.get("url")
                                or image_data.get("imageUrl")
                                or image_data.get("image_url")
                            )
                        else:
                            image_url = image_data  # Might be a string URL
                        
                        if image_url:
                            bt.logging.info(f"Found image URL: {image_url}")
                            # Detect format from URL
                            detected_format = self._detect_format_from_url(image_url)
                            img_bytes = self._download_image(image_url, headers)
                            return img_bytes, detected_format
                    
                    # Fallback to other possible locations
                    images = (
                        status_data.get("generations", [])
                        or status_data.get("generation", {}).get("generated_images", [])
                        or status_data.get("generated_images", [])
                    )
                    
                    if images and len(images) > 0:
                        image_data = images[0]
                        if isinstance(image_data, dict):
                            image_url = (
                                image_data.get("url")
                                or image_data.get("imageUrl")
                                or image_data.get("image_url")
                            )
                        else:
                            image_url = image_data
                        
                        if image_url:
                            detected_format = self._detect_format_from_url(image_url)
                            img_bytes = self._download_image(image_url, headers)
                            return img_bytes, detected_format
                    
                    # Final fallback
                    image_url = (
                        generation_data.get("imageUrl")
                        or generation_data.get("image_url")
                        or generation_data.get("uri")
                        or generation_data.get("url")
                        or status_data.get("imageUrl")
                        or status_data.get("url")
                    )
                    if image_url:
                        detected_format = self._detect_format_from_url(image_url)
                        img_bytes = self._download_image(image_url, headers)
                        return img_bytes, detected_format
                    else:
                        raise RuntimeError(f"Generation completed but no image found: {status_data}")

                elif status_lower in ("failed", "error", "rejected"):
                    error_msg = generation_data.get("error") or generation_data.get("message") or status_data.get("error") or "Unknown error"
                    raise RuntimeError(f"Generation failed: {error_msg}")

                # Still processing
                if attempt % 5 == 0 or attempt == 0:
                    bt.logging.info(f"Generation {generation_id} status: {status} (attempt {attempt + 1}/{max_attempts})")
                time.sleep(poll_interval)

            except requests.exceptions.RequestException as e:
                bt.logging.warning(f"Polling request failed: {e}")
                time.sleep(poll_interval)

        raise RuntimeError(
            f"Generation {generation_id} did not complete within {max_attempts * poll_interval} seconds"
        )

    def _download_image(self, image_url: str, headers: Dict[str, str]) -> bytes:
        """Download image from URL."""
        bt.logging.info(f"Downloading image from: {image_url}")
        
        # Leonardo.ai CDN URLs are typically public and don't need auth headers
        # Use minimal headers to avoid 400 errors
        download_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        
        try:
            response = requests.get(image_url, headers=download_headers, timeout=60, allow_redirects=True)
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            # If download fails, try with authorization header as fallback
            if e.response.status_code in (400, 401, 403):
                bt.logging.warning(f"Download failed, trying with auth headers: {e}")
                download_headers["Authorization"] = headers.get("Authorization", "")
                response = requests.get(image_url, headers=download_headers, timeout=60, allow_redirects=True)
                response.raise_for_status()
                return response.content
            else:
                raise

    def get_service_info(self) -> Dict[str, Any]:
        """Return information about this service."""
        return {
            "name": "Leonardo.ai",
            "type": "api",
            "provider": "cloud.leonardo.ai",
            "available": self.is_available(),
            "supported_tasks": self.get_supported_tasks(),
            "default_model": self.default_model,
        }

