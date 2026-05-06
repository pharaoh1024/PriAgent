"""
Android API knowledge base for RAG-powered semantic verification.

Each entry captures the official Android Developer documentation semantics,
required permissions, risk profile, and common false-positive patterns for
sensitive source/sink APIs.  This structured knowledge lets the SemanticVerifier
reason about APIs without hallucinating — a key advantage over fine-tuning.

Improvement over the paper: entries include `fp_indicators` so few-shot exemplars
can reference concrete patterns, and `risk_level` enables confidence calibration.
"""

from __future__ import annotations
from typing import List, Dict

ANDROID_API_ENTRIES: List[Dict] = [
    # ── Identity & Device ───────────────────────────────────────────────
    {
        "api_name": "android.telephony.TelephonyManager.getDeviceId()",
        "category": "device_identifier",
        "official_purpose": "Returns the unique device ID (IMEI/MEID). Requires READ_PHONE_STATE permission.",
        "permissions_required": ["READ_PHONE_STATE"],
        "risk_level": "high",
        "obsolete_since": "Android 10 (API 29) — returns null for non-privileged apps",
        "typical_use_patterns": [
            "Legacy analytics SDKs use this for device fingerprinting.",
            "MDM/enterprise apps use it for device registration.",
        ],
        "fp_indicators": [
            "Method is called but result is compared to null and short-circuits.",
            "App targets API 29+ and has no privileged permission — call always returns null.",
            "Result is hashed before network transmission — reduces sensitivity.",
        ],
    },
    {
        "api_name": "android.telephony.TelephonyManager.getSubscriberId()",
        "category": "device_identifier",
        "official_purpose": "Returns the IMSI. Requires READ_PHONE_STATE.",
        "permissions_required": ["READ_PHONE_STATE"],
        "risk_level": "high",
        "obsolete_since": "Restricted to carrier-privileged apps on Android 10+",
        "typical_use_patterns": ["Carrier apps for SIM identification."],
        "fp_indicators": [
            "Non-carrier app — will throw SecurityException or return null.",
        ],
    },
    {
        "api_name": "android.provider.Settings.Secure.getString(ANDROID_ID)",
        "category": "device_identifier",
        "official_purpose": "Returns a 64-bit device-unique random ID, reset on factory reset. No special permission needed.",
        "permissions_required": [],
        "risk_level": "medium",
        "obsolete_since": None,
        "typical_use_patterns": [
            "Advertising SDKs as a stable anonymous identifier.",
            "App-specific analytics without linking to user identity.",
        ],
        "fp_indicators": [
            "Value is used only for app-internal session disambiguation.",
            "ID is combined with app-specific namespace to prevent cross-app tracking.",
        ],
    },
    # ── Location ─────────────────────────────────────────────────────────
    {
        "api_name": "android.location.LocationManager.getLastKnownLocation()",
        "category": "location",
        "official_purpose": "Returns the most recently known location. Requires ACCESS_FINE_LOCATION or ACCESS_COARSE_LOCATION.",
        "permissions_required": ["ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION"],
        "risk_level": "high",
        "obsolete_since": None,
        "typical_use_patterns": [
            "Weather apps to fetch local forecast.",
            "Maps/navigation for initial map center.",
            "Delivery apps for address autofill.",
        ],
        "fp_indicators": [
            "Sink is a well-known weather, maps, or forecast API endpoint.",
            "Flow is gated on an explicit user action (button click, 'Use My Location' prompt).",
            "Privacy policy explicitly states location is used for core app functionality.",
        ],
    },
    {
        "api_name": "com.google.android.gms.location.FusedLocationProviderClient.getLastLocation()",
        "category": "location",
        "official_purpose": "Google Play Services fused location — returns best available location. Requires location permissions.",
        "permissions_required": ["ACCESS_FINE_LOCATION"],
        "risk_level": "high",
        "obsolete_since": None,
        "typical_use_patterns": ["Same as LocationManager but via Play Services."],
        "fp_indicators": [
            "Permission runtime check present before call.",
            "Location is only sent to first-party server matching app domain.",
        ],
    },
    # ── Contacts ─────────────────────────────────────────────────────────
    {
        "api_name": "android.provider.ContactsContract.Contacts",
        "category": "contacts",
        "official_purpose": "Content provider for reading device contacts. Requires READ_CONTACTS.",
        "permissions_required": ["READ_CONTACTS"],
        "risk_level": "high",
        "obsolete_since": None,
        "typical_use_patterns": [
            "Messaging apps to resolve phone numbers to names.",
            "Social apps to find friends who also use the service.",
        ],
        "fp_indicators": [
            "Query is limited to a single contact by ID — not a bulk export.",
            "Data stays on-device for display purposes only.",
        ],
    },
    # ── Network / Sinks ───────────────────────────────────────────────────
    {
        "api_name": "okhttp3.OkHttpClient.newCall()",
        "category": "network_sink",
        "official_purpose": "Executes an HTTP request. The actual destination is in the Request URL.",
        "permissions_required": ["INTERNET"],
        "risk_level": "sink",
        "obsolete_since": None,
        "typical_use_patterns": [
            "Any network communication in modern Android apps.",
        ],
        "fp_indicators": [
            "URL pattern matches first-party CDN or known benign service (weather, maps, auth).",
            "Request body does not contain sensitive field names after inspection.",
        ],
    },
    {
        "api_name": "java.net.HttpURLConnection.getOutputStream()",
        "category": "network_sink",
        "official_purpose": "Opens an output stream to send HTTP request body.",
        "permissions_required": ["INTERNET"],
        "risk_level": "sink",
        "obsolete_since": None,
        "typical_use_patterns": ["Legacy HTTP POST calls."],
        "fp_indicators": [
            "Data written is a serialized non-PII object (e.g., crash report with anonymized stack).",
        ],
    },
    {
        "api_name": "android.telephony.SmsManager.sendTextMessage()",
        "category": "sms_sink",
        "official_purpose": "Sends an SMS message. Requires SEND_SMS permission.",
        "permissions_required": ["SEND_SMS"],
        "risk_level": "critical",
        "obsolete_since": None,
        "typical_use_patterns": [
            "Messaging apps for sending user-composed messages.",
            "2FA apps for forwarding verification codes.",
        ],
        "fp_indicators": [
            "User explicitly composed the SMS content in a UI text field.",
            "SMS is a system-level notification app with declared purpose in policy.",
        ],
    },
    # ── Microphone / Camera ───────────────────────────────────────────────
    {
        "api_name": "android.media.MediaRecorder.start()",
        "category": "microphone",
        "official_purpose": "Starts audio/video capture. Requires RECORD_AUDIO and/or CAMERA.",
        "permissions_required": ["RECORD_AUDIO"],
        "risk_level": "critical",
        "obsolete_since": None,
        "typical_use_patterns": [
            "Voice memo apps.",
            "Video calling apps.",
            "Speech-to-text features.",
        ],
        "fp_indicators": [
            "Recording is initiated by explicit user button press.",
            "App category is voice, video, or communication.",
        ],
    },
    # ── Clipboard ─────────────────────────────────────────────────────────
    {
        "api_name": "android.content.ClipboardManager.getPrimaryClip()",
        "category": "clipboard",
        "official_purpose": "Reads clipboard content. On Android 10+ only foreground apps can read.",
        "permissions_required": [],
        "risk_level": "medium",
        "obsolete_since": None,
        "typical_use_patterns": [
            "Password managers to detect copied credentials.",
            "Translation apps to translate copied text.",
        ],
        "fp_indicators": [
            "App is a keyboard or input method service.",
            "Read is triggered by a user 'Paste' action.",
        ],
    },
    # ── Storage ───────────────────────────────────────────────────────────
    {
        "api_name": "android.os.Environment.getExternalStorageDirectory()",
        "category": "file_storage",
        "official_purpose": "Returns the primary shared/external storage directory.",
        "permissions_required": ["READ_EXTERNAL_STORAGE"],
        "risk_level": "medium",
        "obsolete_since": "Deprecated in API 29 — scoped storage preferred",
        "typical_use_patterns": [
            "Media apps to access photos/videos.",
            "File manager apps.",
        ],
        "fp_indicators": [
            "App only reads files it previously wrote (using scoped storage).",
            "Accessed path is the app's own external files directory.",
        ],
    },
]
