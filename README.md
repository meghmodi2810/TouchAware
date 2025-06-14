TouchAware: AI-Powered Good Touch/Bad Touch Detection System
🚀 Detect inappropriate physical contact in real-time using computer vision and machine learning.

🔍 Overview
This application analyzes live video feeds to identify safe vs. inappropriate touch interactions using:

Body pose estimation (MediaPipe)

Hand tracking (OpenCV + MediaPipe)

Distance-based risk assessment (Euclidean distance)

Real-time alerts (visual/audio warnings)

Perfect for child safety, healthcare, and public spaces.

✨ Key Features
✅ Real-Time Detection

Tracks hands approaching sensitive body zones

Classifies interactions as Safe/Warning/Danger

✅ Multi-Input Support

Webcam (any USB camera)

Video files (MP4, AVI)

✅ Alert System
🔊 Audio warnings (customizable sounds)
📝 Incident logging with timestamps

✅ Privacy Protection

Automatic blurring of sensitive areas

No cloud processing (100% local execution)

✅ Machine Learning Mode (Optional)

Train custom detection models

Improve accuracy with user feedback

✅ Web Interface (Optional)

Remote monitoring via Flask server

🛠️ Technical Stack
Category	Technologies
Computer Vision	OpenCV, MediaPipe
GUI	PyQt5
Audio	Pygame, Winsound
ML	Scikit-learn, Joblib
Web	Flask (optional)


TouchAware: AI-Powered Contact Safety Monitor
🛡️ Security Impact & Prevention System
This real-time detection system identifies inappropriate physical interactions using advanced computer vision, creating safer environments for:

🔐 Security Applications
Child Protection

Detects potential grooming behaviors in schools/daycares

Alerts caregivers to inappropriate contact attempts

Healthcare Safety

Monitors patient-staff interactions in hospitals

Prevents abuse of vulnerable patients (elderly/disabled)

Public Space Security

Identifies harassment in public transport

Provides digital evidence for law enforcement

Workplace Compliance

Ensures professional conduct in corporate/industrial settings

Creates audit trails for HR investigations

⚙️ Technical Prevention Features
Instant Audio-Visual Alerts disrupt potential incidents

Privacy-First Design processes all data locally (no cloud storage)

Incident Logging creates timestamped records with:

Body zone involved

Risk level classification

Visual evidence (optional)

