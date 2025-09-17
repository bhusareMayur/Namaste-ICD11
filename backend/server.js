const express = require('express');
const path = require('path');
const cors = require('cors');
const apiRoutes = require('./routes/api');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Set EJS as view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Routes
app.use('/api', apiRoutes);

// Frontend routes
app.get('/', (req, res) => {
  res.render('index');
});

app.get('/search', (req, res) => {
  res.render('search');
});

app.get('/mapping', (req, res) => {
  res.render('mapping');
});

app.get('/analytics', (req, res) => {
  res.render('analytics');
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use((req, res) => {
  res.status(404).render('404');
});

app.listen(PORT, () => {
  console.log(`🚀 NAMASTE-ICD11 Mapping Server running on http://localhost:${PORT}`);
  console.log(`📊 Analytics: http://localhost:${PORT}/analytics`);
  console.log(`🔍 Search: http://localhost:${PORT}/search`);
  console.log(`🗺️  Mapping: http://localhost:${PORT}/mapping`);
});