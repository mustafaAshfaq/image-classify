import base64
from io import BytesIO
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gradio as gr
import torch
from PIL import Image

import app as caption_app
from classification import describe_image, format_detections, get_detector


class ImageDetailsTests(unittest.TestCase):
    @patch("transformers.AutoModelForObjectDetection.from_pretrained")
    @patch("transformers.AutoImageProcessor.from_pretrained")
    @patch("transformers.DetrConfig.get_config_dict")
    def test_detector_normalizes_null_dilation(self, config_loader, processor_loader, model_loader):
        for dilation, expected in [(None, False), (False, False), (True, True)]:
            with self.subTest(dilation=dilation):
                get_detector.cache_clear()
                self.addCleanup(get_detector.cache_clear)
                config_loader.return_value = ({"dilation": dilation, "use_timm_backbone": False}, {})
                processor, model, device = get_detector()
                config = model_loader.call_args.kwargs["config"]
                self.assertIs(config.dilation, expected)
                self.assertEqual(model_loader.call_args.kwargs["revision"], "no_timm")
                self.assertIs(processor, processor_loader.return_value)
                model_loader.return_value.to.assert_called_with(device)
                model.eval.assert_called_once()
                model_loader.reset_mock()

    def test_object_counts_positions_and_annotation(self):
        image = Image.new("RGB", (300, 300), "white")
        detections = {
            "scores": torch.tensor([0.9, 0.8, 0.75]),
            "labels": torch.tensor([1, 1, 2]),
            "boxes": torch.tensor([[0, 0, 60, 60], [240, 240, 300, 300], [120, 120, 180, 180]]),
        }
        annotated, report, rows = format_detections(image, detections, {1: "person", 2: "dog"})
        self.assertIn("**person**: 2", report)
        self.assertIn("**dog**: 1", report)
        self.assertEqual(rows[0], ["person 1", 90.0, "top-left", 4.0])
        self.assertEqual(rows[1][2], "bottom-right")
        self.assertEqual(rows[2][2], "middle-center")
        self.assertNotEqual(annotated.tobytes(), image.tobytes())
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_empty_detections(self):
        detections = {"scores": [], "labels": [], "boxes": []}
        _, report, rows = format_detections(Image.new("RGB", (20, 20)), detections, {})
        self.assertEqual(rows, [])
        self.assertIn("No objects detected", report)

    def test_missing_image(self):
        with self.assertRaises(gr.Error):
            describe_image(None)

    @patch("classification.get_detector")
    @patch("classification.get_captioner")
    def test_analysis_passes_threshold_and_returns_details(self, caption_loader, detector_loader):
        processor = MagicMock()
        processor.return_value.to.return_value = {}
        processor.decode.return_value = "A person with a dog."
        captioner = MagicMock()
        captioner.generate.return_value = [[1]]
        caption_loader.return_value = processor, captioner, "cpu"
        detector = MagicMock()
        detector.config = SimpleNamespace(id2label={1: "person"})
        detection_processor = MagicMock()
        detection_processor.return_value.to.return_value = {}
        detection_processor.post_process_object_detection.return_value = [{
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[10, 10, 50, 50]]),
        }]
        detector_loader.return_value = detection_processor, detector, "cpu"
        annotated, report, rows, caption = describe_image(Image.new("RGBA", (100, 80)), 0.45)
        self.assertEqual(annotated.mode, "RGB")
        self.assertIn("**person**: 1", report)
        self.assertEqual(len(rows), 1)
        self.assertEqual(caption, "A person with a dog.")
        self.assertEqual(
            detection_processor.post_process_object_detection.call_args.kwargs,
            {"threshold": 0.45, "target_sizes": [(80, 100)]},
        )


class AppDataUrlTests(unittest.TestCase):
    @staticmethod
    def _make_data_url(color: str = "blue") -> str:
        buf = BytesIO()
        Image.new("RGBA", (16, 12), color).save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @patch("app.get_captioner")
    def test_describe_image_with_data_url_input(self, caption_loader):
        processor = MagicMock()
        processor.return_value.to.return_value = {}
        processor.decode.return_value = "A blue square."
        model = MagicMock()
        model.generate.return_value = [[1]]
        caption_loader.return_value = processor, model, "cpu"

        result = caption_app.describe_image(None, self._make_data_url())
        self.assertEqual(result, "A blue square.")
        passed_image = processor.call_args.kwargs["images"]
        self.assertEqual(passed_image.mode, "RGB")
        self.assertEqual(passed_image.size, (16, 12))

    @patch("app.get_captioner")
    def test_describe_image_with_direct_data_url_string(self, caption_loader):
        processor = MagicMock()
        processor.return_value.to.return_value = {}
        processor.decode.return_value = "A green square."
        model = MagicMock()
        model.generate.return_value = [[1]]
        caption_loader.return_value = processor, model, "cpu"

        result = caption_app.describe_image(self._make_data_url("green"))
        self.assertEqual(result, "A green square.")

    def test_describe_image_rejects_invalid_data_url(self):
        for invalid in ["https://example.com/img.png", "data:text/plain;base64,aGVsbG8=", "data:image/png;base64,not-base64"]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(gr.Error):
                    caption_app.describe_image(None, invalid)

    def test_describe_image_requires_image_or_data_url(self):
        with self.assertRaises(gr.Error):
            caption_app.describe_image(None, "   ")


if __name__ == "__main__":
    unittest.main()