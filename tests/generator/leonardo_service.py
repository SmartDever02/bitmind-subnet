import os
import traceback
from PIL import Image
import io
import time

from neurons.generator.services.leonardo_service import LeonardoService, Models
from neurons.generator.task_manager import TaskManager

try:
    from gas.verification.c2pa_verification import verify_c2pa
except (ImportError, ModuleNotFoundError, OSError):
    verify_c2pa = None

# Set API key if one isn't already set
os.environ.setdefault(
    "LEONARDO_API_KEY",
    "871466cb-0527-48f3-af20-d4066d3e242a"  # replace with your test key
)

def save_image(img_bytes, filename):
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{filename}"
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    return out_path

def validate_image(img_bytes):
    try:
        Image.open(io.BytesIO(img_bytes)).verify()
        return True
    except Exception:
        return False


def run_model_test(service, manager, model):
    print(f"\n=== Running generation test for model: {model} ===")

    # Use TaskManager.create_task() exactly like in production
    task_id = manager.create_task(
        modality="image",
        prompt="A neon cyberpunk city with flying cars",
        parameters={
            "model": model,
            "format": "png",
            "aspect_ratio": "16:9",
            "seed": 777,
            "negative_prompt": "low quality, blurry"
        },
        webhook_url=None,
        signed_by="test-suite"
    )

    task = manager.get_task(task_id)
    print(f"Task ID: {task_id}")
    print(f"Got task!")

    try:
        start_time = time.time()
        result = service.process(task)
        elapsed = time.time() - start_time
        print(f"Processed task in {elapsed:.2f}s")

        img_bytes = result["data"]
        meta = result["metadata"]
        print(f"Metadata: {meta}")

        print(f"✔ Generated image in {elapsed:.2f}s ({len(img_bytes)/1024:.1f} KB)")
        print(f"✔ Metadata keys: {list(meta.keys())}")

        # Save output
        out = save_image(img_bytes, f"{model.replace('.', '_').replace('-', '_')}.png")
        print(f"✔ Saved to {out}")

        # Validate image integrity
        assert validate_image(img_bytes), "Image failed Pillow validation"

        # C2PA verification (if available)
        if verify_c2pa:
            result_c2pa = verify_c2pa(img_bytes)
            if result_c2pa.verified and result_c2pa.is_trusted_issuer:
                print(f"✅ C2PA verified: {result_c2pa.issuer}")
            else:
                print(f"⚠️  C2PA verification skipped (not available)")
        else:
            print(f"⚠️  C2PA verification skipped (not available)")

        # Validate metadata
        assert meta["model"] == model
        assert meta["provider"] == "leonardo.ai"
        assert meta["format"] in ("PNG", "JPEG", "WEBP")
        assert meta["generation_time"] > 0

        print(f"=== Model {model} PASSED ===")

    except Exception:
        print(f"=== Model {model} FAILED ===")
        print(traceback.format_exc())


def test_invalid_api_key():
    print("\n=== Testing invalid API key ===")
    original_key = os.environ.get("LEONARDO_API_KEY")
    os.environ["LEONARDO_API_KEY"] = "invalid-key"

    service = LeonardoService()
    manager = TaskManager()

    task_id = manager.create_task(
        modality="image",
        prompt="test prompt",
        parameters={"model": Models.PHOENIX},
        webhook_url=None,
        signed_by="test"
    )
    task = manager.get_task(task_id)

    try:
        service.process(task)
        raise AssertionError("❌ Should have failed with invalid API key!")
    except Exception as e:
        print(f"✔ Correctly failed: {e}")
    finally:
        # Restore original key
        if original_key:
            os.environ["LEONARDO_API_KEY"] = original_key
        else:
            os.environ.pop("LEONARDO_API_KEY", None)


def test_invalid_format(service, manager):
    print("\n=== Testing invalid format fallback ===")

    task_id = manager.create_task(
        modality="image",
        prompt="test prompt",
        parameters={"model": Models.PHOENIX, "format": "tiff"},
        webhook_url=None,
        signed_by="test"
    )
    task = manager.get_task(task_id)

    result = service.process(task)
    assert result["metadata"]["format"] == "PNG"
    print("✔ Invalid format 'tiff' auto-corrected to PNG")


def run_full_test_suite():
    print("\n========== Leonardo.ai Full Test Suite ==========\n")

    # Restore real key
    os.environ["LEONARDO_API_KEY"] = os.getenv("LEONARDO_API_KEY")

    service = LeonardoService()
    manager = TaskManager()

    if not service.is_available():
        print("❌ API key missing — cannot run tests")
        return

    # Test all models
    for model in [
        Models.PHOENIX,
        Models.DIFFUSION_XL,
        Models.VISION,
        Models.KINO
    ]:
        run_model_test(service, manager, model)

    # Negative tests
    test_invalid_format(service, manager)
    test_invalid_api_key()

    print("\n========== All Tests Completed ==========\n")


if __name__ == "__main__":
    run_full_test_suite()

