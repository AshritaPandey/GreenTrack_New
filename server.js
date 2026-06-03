require('dotenv').config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const bcrypt = require("bcryptjs");
const mongoose = require("mongoose");
const cloudinary = require("cloudinary").v2;
const { CloudinaryStorage } = require("multer-storage-cloudinary");

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ==========================
// Database Setup
// ==========================
const MONGODB_URI = process.env.MONGODB_URI;

mongoose.connect(MONGODB_URI)
  .then(() => console.log("✅ MongoDB Connected"))
  .catch(err => console.log("❌ MongoDB Error:", err));

// Models
const UserSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true }
});
const User = mongoose.model("User", UserSchema);

const LogSchema = new mongoose.Schema({
  email: { type: String, required: true },
  date: String,
  height: String,
  plant: String,
  notes: String,
  image: String
});
const Log = mongoose.model("Log", LogSchema);

// ==========================
// Cloudinary Setup
// ==========================
const storage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: {
    folder: "greentrack_uploads",
    allowed_formats: ["jpg", "png", "jpeg", "webp"]
  }
});
const upload = multer({ storage });

// ==========================
// Setup Folders and Files (Legacy/Local fallbacks)
// ==========================
const DATA_DIR = process.env.DATA_DIR || __dirname;
const PLANTS_FILE = path.join(DATA_DIR, "plants.json");

function readJSON(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf-8"));
  } catch (err) {
    return {};
  }
}
function writeJSON(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2));
}

if (!fs.existsSync(PLANTS_FILE)) {
  writeJSON(PLANTS_FILE, {});
}
const plantsInit = readJSON(PLANTS_FILE);
if (Object.keys(plantsInit).length === 0) {
  const defaultPlants = {
    "Mango": { soil: "Well-drained alluvial soil", climate: "Tropical to subtropical", temperature: "24°C – 27°C", months: "June – September", watering: "Every 2–3 days" },
    "Banana": { soil: "Rich, well-drained loamy soil", climate: "Tropical", temperature: "26°C – 30°C", months: "June – August", watering: "Daily, especially in summer" },
    "Guava": { soil: "Sandy loam", climate: "Subtropical", temperature: "23°C – 28°C", months: "July – August", watering: "Twice a week" },
    "Papaya": { soil: "Loamy, well-drained", climate: "Tropical", temperature: "22°C – 26°C", months: "June – September", watering: "3 times a week" },
    "Tomato": { soil: "Sandy loam", climate: "Warm, dry", temperature: "21°C – 24°C", months: "October – February", watering: "Every day" },
    "Rose": { soil: "Loamy with good drainage", climate: "Mild and dry", temperature: "15°C – 26°C", months: "October – February", watering: "Every other day" },
    "Marigold": { soil: "Well-drained soil", climate: "Sunny", temperature: "18°C – 30°C", months: "June – October", watering: "2–3 times a week" },
    "Chrysanthemum": { soil: "Sandy loam", climate: "Cool", temperature: "15°C – 20°C", months: "August – October", watering: "Twice a week" },
    "Sunflower": { soil: "Loamy, well-drained", climate: "Warm and sunny", temperature: "20°C – 25°C", months: "February – April", watering: "Once every 2 days" },
    "Jasmine": { soil: "Well-drained rich soil", climate: "Warm", temperature: "20°C – 30°C", months: "June – October", watering: "Every 2 days" },
    "Tulsi": { soil: "Sandy loam with good drainage", climate: "Warm", temperature: "20°C – 35°C", months: "May – July", watering: "Daily" },
    "Aloe Vera": { soil: "Well-drained, sandy soil", climate: "Hot and dry", temperature: "20°C – 30°C", months: "March – June", watering: "Twice a week" },
    "Money Plant": { soil: "Loamy", climate: "Indoor, shaded sunlight", temperature: "15°C – 30°C", months: "Year-round", watering: "Once in 5–7 days" },
    "Hibiscus": { soil: "Loamy and well-drained", climate: "Tropical", temperature: "25°C – 32°C", months: "March – October", watering: "Every day in summer" },
    "Mint": { soil: "Moist, well-drained", climate: "Cool to warm", temperature: "18°C – 24°C", months: "Year-round", watering: "Keep soil moist" },
    "Curry Leaves": { soil: "Well-drained, slightly acidic", climate: "Tropical", temperature: "20°C – 30°C", months: "February – May", watering: "Moderate, once in 2–3 days" },
    "Lemon": { soil: "Sandy loam, well-drained", climate: "Subtropical", temperature: "20°C – 28°C", months: "June – August", watering: "Twice a week" },
    "Ginger": { soil: "Rich, loamy", climate: "Tropical, humid", temperature: "25°C – 30°C", months: "May – June", watering: "Daily" }
  };
  writeJSON(PLANTS_FILE, defaultPlants);
}

// ==========================
// User Authentication Endpoints
// ==========================
app.post("/api/signup", async (req, res) => {
  try {
    const { name, email, password } = req.body;
    if (!name || !email || !password) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    const lowerEmail = email.toLowerCase();
    
    const existingUser = await User.findOne({ email: lowerEmail });
    if (existingUser) {
      return res.status(400).json({ error: "Account already exists" });
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const newUser = new User({
      name,
      email: lowerEmail,
      password: hashedPassword
    });
    await newUser.save();

    res.status(201).json({ message: "Account created successfully", user: { name, email: lowerEmail } });
  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
});

app.post("/api/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    const lowerEmail = email.toLowerCase();
    const user = await User.findOne({ email: lowerEmail });

    if (!user) {
      return res.status(400).json({ error: "Invalid login credentials" });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ error: "Invalid login credentials" });
    }

    res.json({ message: "Login successful", user: { name: user.name, email: user.email } });
  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
});

// ==========================
// Plant Info Endpoints
// ==========================
app.get("/api/plants", (req, res) => {
  const plants = readJSON(PLANTS_FILE);
  res.json(plants);
});

// ==========================
// Tracking & Log Endpoints
// ==========================
app.post("/api/upload", upload.single("image"), async (req, res) => {
  const { email, date, height, notes, plant } = req.body;

  if (!email || !date || !height || !plant) {
    return res.status(400).json({ error: "Missing required fields (email, date, height, plant)" });
  }

  try {
    const newLog = new Log({
      email,
      date,
      height,
      plant,
      notes: notes || "",
      image: req.file ? req.file.path : null // Cloudinary returns URL in path
    });
    
    await newLog.save();

    res.json({
      message: "Plant log saved successfully",
      entry: {
        id: newLog._id.toString(),
        date: newLog.date,
        height: newLog.height,
        plant: newLog.plant,
        notes: newLog.notes,
        image: newLog.image
      }
    });
  } catch(err) {
    console.error(err);
    res.status(500).json({ error: "Failed to save log" });
  }
});

app.get("/api/tracker/:email", async (req, res) => {
  try {
    const userLogs = await Log.find({ email: req.params.email });
    const formattedLogs = userLogs.map(log => ({
      id: log._id.toString(),
      date: log.date,
      height: log.height,
      plant: log.plant,
      notes: log.notes,
      image: log.image
    }));
    res.json(formattedLogs);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch logs" });
  }
});

app.post("/api/delete-log", async (req, res) => {
  const { email, id } = req.body;
  if (!email || !id) {
    return res.status(400).json({ error: "Missing email or log id" });
  }

  try {
    const deletedLog = await Log.findOneAndDelete({ _id: id, email });
    if (deletedLog) {
      res.json({ message: "Log deleted successfully" });
    } else {
      res.status(404).json({ error: "No logs found for this user or unauthorized" });
    }
  } catch (err) {
    res.status(500).json({ error: "Failed to delete log" });
  }
});

app.post("/api/analyze-health", async (req, res) => {
  const { plant, notes, imageUrl } = req.body;
  
  try {
    const formData = new URLSearchParams();
    formData.append('plant', plant);
    formData.append('notes', notes || '');
    
    if (imageUrl) {
        if (imageUrl.startsWith('http')) {
            formData.append('image_url', imageUrl);
        } else {
            const fullImagePath = path.join(DATA_DIR, imageUrl);
            formData.append('image_path', fullImagePath);
        }
    }

    const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "https://greentrack-ai-v2.onrender.com";
    const response = await fetch(`${ML_SERVICE_URL}/predict-disease`, {
      method: "POST",
      body: formData
    });
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error("ML Service Error:", error);
    res.status(500).json({ error: "ML Service unavailable" });
  }
});

app.post("/api/recommend", async (req, res) => {
  const { month, current_temp, soil_type, area, rain_prob, nutrients } = req.body;
  
  try {
    const MONGODB_URI = process.env.MONGODB_URI;
    const CLOUDINARY_URL = process.env.CLOUDINARY_URL;
    // Force connection to live AI engine if not running locally with explicit env var
    const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "https://greentrack-ai-v2.onrender.com";
    const response = await fetch(`${ML_SERVICE_URL}/recommend-plant`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ month, current_temp, soil_type, area, rain_prob, nutrients })
    });
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error("ML Service Error:", error);
    res.status(500).json({ error: "ML Service unavailable" });
  }
});

// ==========================
// Static File Serving
// ==========================
app.use(express.static(__dirname));

// Explicitly serve index.html as the root
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});
