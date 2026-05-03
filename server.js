const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const bcrypt = require("bcryptjs");

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ==========================
// Setup Folders and Files
// ==========================
// DATA_DIR allows Render to use a persistent disk mount
const DATA_DIR = process.env.DATA_DIR || __dirname;

const uploadDir = path.join(DATA_DIR, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const USERS_FILE = path.join(DATA_DIR, "users.json");
const LOGS_FILE = path.join(DATA_DIR, "logs.json");
const PLANTS_FILE = path.join(DATA_DIR, "plants.json");

// Helper to ensure files exist
[USERS_FILE, LOGS_FILE, PLANTS_FILE].forEach(file => {
  if (!fs.existsSync(file)) {
    fs.writeFileSync(file, JSON.stringify({}, null, 2));
  }
});

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

// Populate default plants if empty
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
// Multer setup (image upload)
// ==========================
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, Date.now() + "-" + file.originalname)
});
const upload = multer({ storage });

// ==========================
// User Authentication Endpoints
// ==========================
app.post("/api/signup", async (req, res) => {
  try {
    const { name, email, password } = req.body;
    if (!name || !email || !password) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    const users = readJSON(USERS_FILE);
    const lowerEmail = email.toLowerCase();
    
    if (users[lowerEmail]) {
      return res.status(400).json({ error: "Account already exists" });
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    users[lowerEmail] = {
      name,
      email: lowerEmail,
      password: hashedPassword
    };

    writeJSON(USERS_FILE, users);
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

    const users = readJSON(USERS_FILE);
    const lowerEmail = email.toLowerCase();
    const user = users[lowerEmail];

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
app.post("/api/upload", upload.single("image"), (req, res) => {
  const { email, date, height, notes, plant } = req.body;

  if (!email || !date || !height || !plant) {
    return res.status(400).json({ error: "Missing required fields (email, date, height, plant)" });
  }

  const logs = readJSON(LOGS_FILE);
  if (!logs[email]) logs[email] = [];

  const entry = {
    id: Date.now().toString(),
    date,
    height,
    plant,
    notes: notes || "",
    image: req.file ? `/uploads/${req.file.filename}` : null
  };

  logs[email].push(entry);
  writeJSON(LOGS_FILE, logs);

  res.json({
    message: "Plant log saved successfully",
    entry
  });
});

app.get("/api/tracker/:email", (req, res) => {
  const logs = readJSON(LOGS_FILE);
  const userLogs = logs[req.params.email] || [];
  
  // Migration fallback: assign IDs if missing
  let changed = false;
  userLogs.forEach((log, index) => {
    if (!log.id) {
      log.id = `migrated-${Date.now()}-${index}`;
      changed = true;
    }
  });
  
  if (changed) {
    logs[req.params.email] = userLogs;
    writeJSON(LOGS_FILE, logs);
  }

  res.json(userLogs);
});

app.post("/api/delete-log", (req, res) => {
  const { email, id } = req.body;
  if (!email || !id) {
    return res.status(400).json({ error: "Missing email or log id" });
  }

  const logs = readJSON(LOGS_FILE);
  if (logs[email]) {
    logs[email] = logs[email].filter(log => log.id !== id);
    writeJSON(LOGS_FILE, logs);
    res.json({ message: "Log deleted successfully" });
  } else {
    res.status(404).json({ error: "No logs found for this user" });
  }
});

app.post("/api/analyze-health", async (req, res) => {
  const { plant, notes, imageUrl } = req.body;
  
  try {
    const fullImagePath = imageUrl ? path.join(DATA_DIR, imageUrl.replace('http://localhost:3000', '')) : null;
    
    const formData = new URLSearchParams();
    formData.append('plant', plant);
    formData.append('notes', notes || '');
    if (fullImagePath) formData.append('image_path', fullImagePath);

    const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";
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
    const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";
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
app.use("/uploads", express.static(uploadDir));
app.use(express.static(__dirname));

// Explicitly serve index.html as the root
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
