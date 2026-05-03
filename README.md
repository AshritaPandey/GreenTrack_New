# 🌱 GreenTrack – Setup and Run Instructions

Follow these simple steps to get your Digital Plant Growth Tracking platform up and running on your local machine.

## Prerequisites
- **Node.js:** Ensure you have Node.js installed. You can download it from [nodejs.org](https://nodejs.org/).

---

## 🚀 Getting Started

### 1. Install Dependencies
Open your terminal (Command Prompt or PowerShell on Windows) in the project directory and run:

```bash
npm install
```
This will install all necessary packages, including **Express**, **Cors**, **Multer** for image uploads, and **Bcryptjs** for secure logins.

### 2. Start the Backend Server
Once the installation is complete, start the server by running:

```bash
node server.js
```
You should see a message: `✅ Server running on http://localhost:3000`.

### 3. Access the Application
Open your web browser and navigate to the following URL:

```text
http://localhost:3000/index.html
```

---

## 🛠 Features Summary
- **Premium Design:** Full glassmorphism UI with smooth animations.
- **Growth Tracker:** Visualize your plant's progress with dynamic **Chart.js** trends.
- **Smart Health Analysis:** Click "Run Health Check" on your logs to see species-specific expert suggestions.
- **Plant Guidance:** Explore the "Plant Selection" guide to learn about soil, climate, and watering requirements.
- **Secure Data:** All your progress is saved automatically in local JSON files.

## 📁 Project Structure
- `index.html`: Welcome & Auth page.
- `main.html`: Dashboard & Plant Selection guide.
- `tracker.html`: Growth logs and Chart.js integration.
- `server.js`: The Node.js API and storage logic.
- `style.css`: The central design system.

Enjoy your green journey! 🌿
