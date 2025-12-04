# 📱 PokerVision Mobile App Deployment Guide

Your React web app has been successfully converted to a mobile app using Capacitor! This guide will walk you through publishing to both the Apple App Store and Google Play Store.

## ✅ What's Been Done

- ✅ Capacitor installed and configured
- ✅ iOS project created (`frontend/ios/`)
- ✅ Android project created (`frontend/android/`)
- ✅ Camera integration added for native photo capture
- ✅ YOLOv11 model upgrade committed
- ✅ Build scripts added to package.json

## 🎯 Your Web App is Still Live!

Your web app (pokervision.netlify.app) will continue to work as before. The mobile app is an additional platform.

---

## 📋 Prerequisites Checklist

### For iOS (Apple App Store)
- [ ] Mac computer with macOS (required for iOS development)
- [ ] Xcode installed (free from Mac App Store)
- [ ] Apple Developer Account ($99/year) - [Sign up here](https://developer.apple.com/programs/)
- [ ] iPhone or iPad for testing (optional but recommended)

### For Android (Google Play Store)
- [ ] Google Play Developer Account ($25 one-time) - [Sign up here](https://play.google.com/console/signup)
- [ ] Android Studio installed (free) - [Download here](https://developer.android.com/studio)
- [ ] Android device for testing (optional)

### Required for Both
- [ ] Privacy Policy URL (required by both app stores)
  - You can generate one free at: https://www.freeprivacypolicy.com
  - Or use: https://app-privacy-policy-generator.nisrulz.com/
- [ ] App icons (1024x1024 PNG)
- [ ] App screenshots (see requirements below)
- [ ] App description and marketing materials

---

## 📱 Part 1: iOS App Store Deployment

### Step 1: Set Up Development Environment

1. **Install Xcode** (if not already installed):
   ```bash
   # Open Mac App Store and search for "Xcode"
   # Or download from https://developer.apple.com/xcode/
   ```

2. **Open your iOS project**:
   ```bash
   cd frontend
   npm run cap:open:ios
   # This opens the project in Xcode
   ```

### Step 2: Configure App in Xcode

1. **Select the project** in the left sidebar (App icon)
2. **Update these settings** in the General tab:
   - **Display Name**: PokerVision
   - **Bundle Identifier**: `com.pokervision.app` (or your custom domain)
   - **Version**: 1.0.0
   - **Build**: 1

3. **Add Camera Permissions**:
   - Go to `Info.plist` in the left sidebar
   - Add these keys:
     ```xml
     NSCameraUsageDescription: "PokerVision needs camera access to scan poker tables"
     NSPhotoLibraryUsageDescription: "PokerVision needs photo access to analyze poker images"
     ```

### Step 3: Create App Icons

1. **Prepare your icon**:
   - Size: 1024x1024 pixels
   - Format: PNG without transparency
   - No rounded corners (iOS adds them automatically)

2. **Add to Xcode**:
   - Click "AppIcon" in Assets.xcassets
   - Drag your 1024x1024 icon to the "App Store iOS" slot
   - Xcode will generate all required sizes

### Step 4: Test on Simulator or Device

1. **Select a simulator** from the top toolbar (e.g., "iPhone 15 Pro")
2. **Click the Play button** (▶️) to build and run
3. **Test camera functionality** (note: camera won't work in simulator)

**To test on real iPhone:**
1. Connect iPhone via USB
2. Select your iPhone from device dropdown
3. Click "Run" (you may need to trust your developer certificate on the iPhone)

### Step 5: Create App Store Connect Listing

1. **Log into App Store Connect**: https://appstoreconnect.apple.com
2. **Click "My Apps" → "+" → "New App"**
3. **Fill in details**:
   - **Platforms**: iOS
   - **Name**: PokerVision
   - **Primary Language**: English
   - **Bundle ID**: Select `com.pokervision.app`
   - **SKU**: pokervision-001 (unique identifier for your records)

4. **Complete app information**:
   - **App Subtitle**: "AI-Powered Poker Hand Analyzer"
   - **Category**: Entertainment or Games
   - **Privacy Policy URL**: (your privacy policy URL)
   - **Support URL**: (your website or GitHub)

### Step 6: Prepare Screenshots

**Required screenshot sizes** (use iPhone simulator):
- 6.7" display (iPhone 15 Pro Max): 1290 x 2796 pixels
- 6.5" display (iPhone 14 Plus): 1284 x 2778 pixels

**How to take screenshots:**
1. Run app in Xcode simulator
2. Navigate to main screens
3. Press Cmd+S to save screenshot
4. Upload 3-5 screenshots showing:
   - Landing page
   - Upload/camera interface
   - Results page
   - Card correction modal

### Step 7: Archive and Upload to App Store

1. **In Xcode, select "Any iOS Device" from device dropdown**
2. **Product → Archive** (wait for build to complete)
3. **When archive window appears, click "Distribute App"**
4. **Select "App Store Connect" → Next**
5. **Select "Upload" → Next**
6. **Follow the wizard, accepting defaults**
7. **Wait for upload to complete** (5-10 minutes)

### Step 8: Submit for Review

1. **Return to App Store Connect**
2. **Go to your app → "1.0 Prepare for Submission"**
3. **Add your screenshots and description**
4. **Fill in App Review Information**:
   - Contact info (your email/phone)
   - Demo account (if needed - not needed for PokerVision)
5. **Content Rights**: Check the box confirming you have rights
6. **Click "Submit for Review"**

**Review timeline**: 1-3 days typically

---

## 🤖 Part 2: Google Play Store Deployment

### Step 1: Set Up Android Studio

1. **Install Android Studio**: https://developer.android.com/studio
2. **Open Android Studio**
3. **File → Open** → Select `frontend/android/` folder
4. **Wait for Gradle sync** (first time takes 5-10 minutes)

### Step 2: Configure Android App

1. **Update `android/app/build.gradle`**:
   ```gradle
   android {
       defaultConfig {
           applicationId "com.pokervision.app"
           minSdkVersion 24
           targetSdkVersion 34
           versionCode 1
           versionName "1.0.0"
       }
   }
   ```

2. **Add Camera Permissions** (already done by Capacitor):
   - Check `android/app/src/main/AndroidManifest.xml`
   - Should have:
     ```xml
     <uses-permission android:name="android.permission.CAMERA" />
     <uses-permission android:name="android.permission.INTERNET" />
     ```

### Step 3: Create App Icon

1. **Right-click `res` folder** in Android Studio
2. **New → Image Asset**
3. **Upload your 1024x1024 icon**
4. **Click "Next" → "Finish"**

### Step 4: Test on Emulator or Device

1. **Click the "Play" button** (▶️) in Android Studio
2. **Select emulator or connected device**
3. **Test camera and upload functionality**

### Step 5: Generate Signed Release APK/AAB

1. **In Android Studio: Build → Generate Signed Bundle / APK**
2. **Select "Android App Bundle" (AAB)** - required for Play Store
3. **Create new keystore**:
   - Click "Create new..."
   - Key store path: Choose a secure location (SAVE THIS FILE!)
   - Password: Create strong password (SAVE THIS PASSWORD!)
   - Alias: pokervision-key
   - Valid for: 25 years
   - Fill in your name/organization

   **⚠️ CRITICAL: Backup your keystore file! You cannot update your app without it!**

4. **Select "release" build variant**
5. **Click "Create"**
6. **Wait for build** (AAB file will be in `android/app/release/`)

### Step 6: Create Google Play Console Listing

1. **Log into Google Play Console**: https://play.google.com/console
2. **Click "Create app"**
3. **Fill in details**:
   - **App name**: PokerVision
   - **Default language**: English
   - **App or game**: App
   - **Free or paid**: Free

4. **Complete store listing**:
   - **Short description** (80 chars):
     "Instantly analyze poker hands using AI-powered card detection"

   - **Full description** (4000 chars):
     ```
     🃏 PokerVision - AI-Powered Poker Hand Analyzer

     Instantly analyze Texas Hold'em poker situations with advanced computer vision!
     Simply snap a photo of your poker table and get instant hand rankings,
     winner predictions, and detailed analysis.

     ✨ Features:
     • 📸 Quick photo capture or upload
     • 🤖 AI-powered card detection (YOLOv11)
     • 🎯 Automatic hand ranking
     • 🏆 Winner determination
     • ✏️ Manual card correction
     • 📊 Detailed probability analysis

     Perfect for:
     • Learning poker hand rankings
     • Settling friendly game disputes
     • Analyzing home game situations
     • Poker education

     How it works:
     1. Take a photo of your poker table
     2. AI automatically detects all cards
     3. Get instant hand rankings and winner

     Privacy: All image processing happens on our secure servers.
     No data is stored or shared.
     ```

### Step 7: Prepare Screenshots

**Required sizes** (use Android emulator):
- Phone: 1080 x 1920 to 1080 x 2340 pixels
- 7" Tablet: 1600 x 2560 pixels (optional)
- 10" Tablet: 2048 x 2732 pixels (optional)

**Minimum required**: 2 screenshots per device type

### Step 8: Upload AAB and Publish

1. **Go to "Production" in left sidebar**
2. **Click "Create new release"**
3. **Upload your AAB file** (from android/app/release/)
4. **Fill in release notes**:
   ```
   Initial release of PokerVision!
   • AI-powered poker hand detection
   • Instant winner determination
   • Camera and file upload support
   ```

5. **Content rating questionnaire**:
   - Select "Continue"
   - Answer questions (PokerVision has no violence, adult content, etc.)

6. **Target audience**: Select age 13+ (or 18+ if targeting gambling)

7. **Review and publish**:
   - Review all sections
   - Click "Send for review"

**Review timeline**: Usually 24-48 hours

---

## 🔧 Testing & Development Commands

### Build and test web version
```bash
cd frontend
npm run build
npm run preview
```

### Sync changes to native apps
```bash
npm run cap:sync
```

### Open in Xcode (iOS)
```bash
npm run cap:open:ios
```

### Open in Android Studio
```bash
npm run cap:open:android
```

### Run on iOS device/simulator
```bash
npm run cap:run:ios
```

### Run on Android device/emulator
```bash
npm run cap:run:android
```

---

## 📸 Camera Integration Features

Your app now has native camera support!

**On mobile devices:**
- ✅ "Take Photo" button opens native camera
- ✅ "Choose File" button opens photo gallery
- ✅ Images are processed and sent to your backend

**On web:**
- ✅ Drag-and-drop still works
- ✅ File picker still works
- ✅ No camera button shown (desktop doesn't have cameras)

---

## 🔐 Privacy Policy Requirements

Both app stores require a privacy policy. Here's a template:

### What to include:
1. **Data Collection**:
   - "We process poker table images to detect cards"
   - "Images are sent to our server for AI analysis"
   - "We do not store images after processing"

2. **Third-party Services**:
   - "We use Hugging Face Spaces for ML inference"

3. **Camera/Photos**:
   - "Camera is used only when you tap 'Take Photo'"
   - "Photos are used only for card detection"

### Generate Privacy Policy:
- https://www.freeprivacypolicy.com
- https://app-privacy-policy-generator.nisrulz.com/

---

## 💰 Cost Breakdown

### One-time Costs
- Google Play Developer Account: **$25** (one-time)

### Annual Costs
- Apple Developer Program: **$99/year**

### Ongoing Costs
- Hugging Face Spaces (backend):
  - Free tier: Limited usage
  - Paid tier: ~$50-200/month for production

### Recommendations:
1. **Start with free backend** tier to test the waters
2. **Monitor usage** in first month
3. **Upgrade backend** if you get 100+ daily users
4. **Consider migrating** to AWS/Railway if costs get high

---

## 🚀 Next Steps After Publishing

### Week 1: Initial Launch
- [ ] Share app link with friends/family for testing
- [ ] Monitor crash reports in App Store Connect / Play Console
- [ ] Gather user feedback

### Week 2-4: Iteration
- [ ] Fix any critical bugs
- [ ] Add user-requested features
- [ ] Optimize backend performance

### Month 2+: Growth
- [ ] Add app analytics (Google Analytics, Mixpanel)
- [ ] Create app preview video for store listings
- [ ] Submit to app review sites
- [ ] Build social media presence

---

## 🆘 Troubleshooting

### iOS Build Fails
- Ensure Xcode is up to date
- Clean build folder: Product → Clean Build Folder
- Delete DerivedData: `rm -rf ~/Library/Developer/Xcode/DerivedData`

### Android Build Fails
- Invalidate caches: File → Invalidate Caches / Restart
- Update Gradle: Accept updates in bottom-right corner
- Check Java version: `java --version` (need Java 17+)

### Camera Not Working
- Check permissions in device settings
- Verify `capacitor.config.json` has Camera plugin
- Run `npx cap sync` after any config changes

### Backend Connection Issues
- Verify backend URL in `frontend/src/services/api.js`
- Check CORS settings on backend
- Test backend health endpoint

---

## 📚 Additional Resources

### Official Docs
- [Capacitor Docs](https://capacitorjs.com/docs)
- [App Store Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [Google Play Policy](https://play.google.com/about/developer-content-policy/)

### Asset Generators
- [App Icon Generator](https://www.appicon.co/)
- [Screenshot Mockup Generator](https://www.screely.com/)
- [Privacy Policy Generator](https://www.freeprivacypolicy.com/)

### Communities
- [r/iOSProgramming](https://reddit.com/r/iOSProgramming)
- [r/androiddev](https://reddit.com/r/androiddev)
- [Capacitor Discord](https://discord.com/invite/UPYYRhtyzp)

---

## 🎉 You're Ready!

Your app is now mobile-ready! The setup is complete and you can:
1. ✅ Test on iOS/Android devices
2. ✅ Submit to app stores
3. ✅ Keep your web app running simultaneously

Good luck with your app launch! 🚀
