"""
mattress/basler_camera.py -- wrapper around the Basler acA4600-10uc (pypylon)
for the side view (label + QR).

This camera is USB3 Vision, driven by Basler's Pylon SDK -- it does NOT go
through cv2.VideoCapture/V4L2 like the webcams. Everything downstream (QRReader,
LabelOCRReader in claim.py) is unaffected: they only ever consume a BGR numpy
array, regardless of which camera produced it. This module's only job is to
produce that array.

NOT YET VERIFIED ON REAL HARDWARE -- I do not have this camera or the Pylon
SDK available to execute-test this. It is grounded in Basler's own official
pypylon samples (TlFactory -> InstantCamera -> GrabOne/RetrieveResult ->
ImageFormatConverter), syntax-checked, but not run. Verify with
test_basler_camera.py (companion script) on the Pi before wiring this into
reconcile_live.py -- same discipline used for the texture and QR adapters.

Open items to confirm on real hardware, not assumed here:
  - Actual sensor resolution (read camera.Width.Max / camera.Height.Max at
    runtime rather than trust a number from the model name).
  - Whether default PixelFormat needs to be set explicitly (some Basler
    color cameras default to Bayer/raw; the ImageFormatConverter handles
    the conversion to BGR8 regardless, but a wrong sensor PixelFormat can
    still affect image quality -- check pylon-viewer's default first).
  - Grab timeout tuning (GRAB_TIMEOUT_MS below is a starting guess, not
    measured).
"""

from typing import Optional
import numpy as np

try:
    from pypylon import pylon
except ImportError as e:
    raise ImportError(
        "pypylon is not installed. Run: pip install pypylon "
        "(and ensure the Pylon SDK + udev rules are installed on the Pi -- "
        "see setup instructions)."
    ) from e


GRAB_TIMEOUT_MS = 2000   # placeholder -- tune once you see real grab latency


class BaslerSideCamera:
    """Single-camera wrapper: open once, grab frames as needed, close explicitly.

    Use as a context manager so the device is always released cleanly, the
    same discipline as the existing webcam grab() function (single-owner
    USB, must release before another process can use it):

        with BaslerSideCamera(serial="22424917") as cam:
            frame = cam.grab_frame()
    """

    def __init__(self, serial: Optional[str] = None):
        """serial: optional serial number to open a SPECIFIC device (recommended
        once more than one Basler camera might ever be attached). Read off the
        camera's own label if not already known. If None, opens the first
        device the Pylon driver finds -- fine for a single-camera setup, but
        an explicit serial is safer and is what CreateFirstDevice() implicitly
        skips."""
        self._serial = serial
        self._camera = None
        self._converter = None

    def open(self):
        tl_factory = pylon.TlFactory.GetInstance()

        if self._serial:
            # EnumerateDevices() returns DeviceInfo objects; find the one
            # matching our serial, then turn it into an actual device handle.
            devices = tl_factory.EnumerateDevices()
            matches = [d for d in devices if d.GetSerialNumber() == self._serial]
            if not matches:
                found = [d.GetSerialNumber() for d in devices]
                raise RuntimeError(
                    f"No Basler camera with serial {self._serial!r} found. "
                    f"Devices seen: {found!r}"
                )
            pylon_device = tl_factory.CreateDevice(matches[0])
        else:
            # No serial given -- open whichever device the driver finds first.
            # Raises pylon.RuntimeException("No devices are available.") if
            # none are attached -- correct fail-loud behaviour, not caught here.
            pylon_device = tl_factory.CreateFirstDevice()

        self._camera = pylon.InstantCamera(pylon_device)
        self._camera.Open()

        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        return self

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def grab_frame(self) -> Optional[np.ndarray]:
        """Grab exactly one frame. Returns a BGR numpy array, or None on a
        failed/timed-out grab (fails loud in the caller's control, never
        raises for a routine miss -- mirrors the webcam grab()'s style of
        returning None rather than crashing the whole reconcile cycle)."""
        if self._camera is None or not self._camera.IsOpen():
            raise RuntimeError("camera not open -- call open() or use as a context manager")

        result = self._camera.GrabOne(GRAB_TIMEOUT_MS)
        try:
            if not result.GrabSucceeded():
                return None
            image = self._converter.Convert(result)
            return image.GetArray()   # BGR8, shape (H, W, 3) -- same shape cv2 frames use
        finally:
            result.Release()

    def close(self):
        if self._camera is not None:
            if self._camera.IsOpen():
                self._camera.Close()
            self._camera = None


def grab_burst_until_decoded(qr_reader, serial: Optional[str] = None,
                             attempts: int = 8) -> tuple:
    """Basler equivalent of reconcile_live.py's grab_burst_until_decoded().

    Opens the camera ONCE for the whole burst (much cheaper than the webcam
    version's per-attempt open/close, since Basler grabs are fast once the
    device is open), tries up to `attempts` single-shot grabs, returns the
    first frame whose QR decodes.

    Falls back to returning the LAST captured frame (decoded or not) so the
    caller still has something to show/log/OCR even on a full-burst miss --
    same fallback contract as the webcam version.

    Returns: (frame_bgr_or_None, qr_claim_or_None)

    NOT YET VERIFIED -- run test_basler_camera.py first. In particular: is
    a fresh GrabOne() per attempt sufficient, or does this camera need
    StartGrabbing()/RetrieveResult() continuous-mode to get fresh frames at
    useful speed? Single-shot GrabOne re-arms the trigger each call per
    Basler's samples, so this should be correct, but burst timing has not
    been measured on real hardware.
    """
    last_frame = None
    with BaslerSideCamera(serial=serial) as cam:
        for _ in range(attempts):
            frame = cam.grab_frame()
            if frame is None:
                continue
            last_frame = frame
            claim = qr_reader.read(frame)
            if claim is not None:
                return frame, claim
    return last_frame, None