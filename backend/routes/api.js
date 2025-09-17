const express = require('express');
const axios = require('axios');
const Fuse = require('fuse.js');
const csvLoader = require('../utils/csvLoader');

const router = express.Router();

// In-memory data storage
let dataCache = {
  namaste: [],
  icd11: [],
  mappings: [],
  lastLoaded: null
};

// ML service configuration
const ML_SERVICE_URL = 'http://127.0.0.1:5001';

// Initialize data on startup
async function initializeData() {
  try {
    const data = await csvLoader.loadAllData();
    dataCache = {
      ...data,
      lastLoaded: new Date()
    };
  } catch (error) {
    console.error('❌ Error initializing data:', error);
  }
}

// Initialize data when module loads
initializeData();

// Configure Fuse.js for fuzzy search
const fuseOptions = {
  includeScore: true,
  threshold: 0.4, // 0.0 = exact match, 1.0 = match anything
  keys: []
};

// Health check endpoint
router.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    dataLoaded: dataCache.lastLoaded,
    counts: {
      namaste: dataCache.namaste.length,
      icd11: dataCache.icd11.length,
      mappings: dataCache.mappings.length
    }
  });
});

// Get all NAMASTE entries
router.get('/namaste', (req, res) => {
  res.json({
    count: dataCache.namaste.length,
    data: dataCache.namaste
  });
});

// Get all ICD-11 entries
router.get('/icd11', (req, res) => {
  res.json({
    count: dataCache.icd11.length,
    data: dataCache.icd11
  });
});

// Get all mappings
router.get('/mappings', (req, res) => {
  res.json({
    count: dataCache.mappings.length,
    data: dataCache.mappings
  });
});

// Search endpoint with fuzzy matching
router.get('/search', (req, res) => {
  const { q: query, type } = req.query;
  
  if (!query || !type) {
    return res.status(400).json({ 
      error: 'Query (q) and type parameters are required',
      usage: '/api/search?q=fever&type=namaste'
    });
  }

  let dataset, searchKeys;
  
  switch (type.toLowerCase()) {
    case 'namaste':
      dataset = dataCache.namaste;
      searchKeys = ['Traditional_Term', 'English_Translation', 'NAMASTE_Code', 'Category'];
      break;
    case 'icd':
    case 'icd11':
      dataset = dataCache.icd11;
      searchKeys = ['Title', 'ICD11_Code', 'Category'];
      break;
    default:
      return res.status(400).json({ 
        error: 'Invalid type parameter',
        validTypes: ['namaste', 'icd11']
      });
  }

  // Configure Fuse for this search
  const fuse = new Fuse(dataset, {
    ...fuseOptions,
    keys: searchKeys
  });

  const results = fuse.search(query);
  
  res.json({
    query,
    type,
    count: results.length,
    results: results.map(result => ({
      item: result.item,
      score: result.score
    }))
  });
});

// Autocomplete endpoint
router.get('/autocomplete', (req, res) => {
  const { q: query, type, limit = 10 } = req.query;
  
  if (!query || !type) {
    return res.status(400).json({ 
      error: 'Query (q) and type parameters are required',
      usage: '/api/autocomplete?q=jwa&type=namaste&limit=5'
    });
  }

  let dataset, searchKeys;
  
  switch (type.toLowerCase()) {
    case 'namaste':
      dataset = dataCache.namaste;
      searchKeys = ['Traditional_Term', 'English_Translation'];
      break;
    case 'icd':
    case 'icd11':
      dataset = dataCache.icd11;
      searchKeys = ['Title'];
      break;
    default:
      return res.status(400).json({ 
        error: 'Invalid type parameter',
        validTypes: ['namaste', 'icd11']
      });
  }

  // More lenient fuzzy search for autocomplete
  const fuse = new Fuse(dataset, {
    includeScore: true,
    threshold: 0.6,
    keys: searchKeys
  });

  const results = fuse.search(query).slice(0, parseInt(limit));
  
  res.json({
    query,
    type,
    suggestions: results.map(result => ({
      ...result.item,
      score: result.score
    }))
  });
});

// Get mapping for specific NAMASTE code
router.get('/map/:namasteCode', (req, res) => {
  const { namasteCode } = req.params;
  
  // Find mapping
  const mapping = dataCache.mappings.find(m => m.NAMASTE_Code === namasteCode);
  
  if (!mapping) {
    return res.json({
      namasteCode,
      mapping: null,
      message: 'No mapping found for this NAMASTE code'
    });
  }

  // Get related NAMASTE entry
  const namasteEntry = dataCache.namaste.find(n => n.NAMASTE_Code === namasteCode);
  
  // Get related ICD-11 entries
  const icd11TM2 = dataCache.icd11.find(i => i.ICD11_Code === mapping.ICD11_TM2_Code);
  const icd11Bio = dataCache.icd11.find(i => i.ICD11_Code === mapping.ICD11_Biomedicine_Code);

  res.json({
    namasteCode,
    namasteEntry,
    mapping,
    icd11Entries: {
      traditional: icd11TM2,
      biomedicine: icd11Bio
    }
  });
});

// Generate FHIR-like Condition resource
router.post('/generate-fhir', async (req, res) => {
  const { patientId, namasteCode, clinician } = req.body;
  
  if (!patientId || !namasteCode || !clinician) {
    return res.status(400).json({ 
      error: 'patientId, namasteCode, and clinician are required' 
    });
  }

  // Get NAMASTE entry
  const namasteEntry = dataCache.namaste.find(n => n.NAMASTE_Code === namasteCode);
  if (!namasteEntry) {
    return res.status(404).json({ 
      error: 'NAMASTE code not found',
      code: namasteCode 
    });
  }

  // Check for existing mapping
  const mapping = dataCache.mappings.find(m => m.NAMASTE_Code === namasteCode);
  let icd11Codes = [];
  let mlSuggestion = null;

  if (mapping) {
    // Use existing mapping
    const icd11TM2 = dataCache.icd11.find(i => i.ICD11_Code === mapping.ICD11_TM2_Code);
    const icd11Bio = dataCache.icd11.find(i => i.ICD11_Code === mapping.ICD11_Biomedicine_Code);
    
    if (icd11TM2) icd11Codes.push(icd11TM2);
    if (icd11Bio) icd11Codes.push(icd11Bio);
  } else {
    // Try to get ML suggestion
    try {
      const mlResponse = await axios.get(`${ML_SERVICE_URL}/predict`, {
        params: { text: namasteEntry.English_Translation },
        timeout: 5000
      });
      
      if (mlResponse.data && mlResponse.data.length > 0) {
        mlSuggestion = mlResponse.data[0]; // Top suggestion
        // Find full ICD entry
        const suggestedIcd = dataCache.icd11.find(i => i.ICD11_Code === mlSuggestion.icd_code);
        if (suggestedIcd) {
          icd11Codes.push(suggestedIcd);
        }
      }
    } catch (error) {
      console.warn('⚠️  ML service unavailable:', error.message);
    }
  }

  // Generate FHIR-like Condition resource
  const fhirCondition = {
    resourceType: "Condition",
    id: `condition-${patientId}-${Date.now()}`,
    meta: {
      profile: ["http://hl7.org/fhir/StructureDefinition/Condition"],
      lastUpdated: new Date().toISOString()
    },
    status: "active",
    category: [{
      coding: [{
        system: "http://terminology.hl7.org/CodeSystem/condition-category",
        code: "encounter-diagnosis",
        display: "Encounter Diagnosis"
      }]
    }],
    code: {
      coding: [
        // NAMASTE coding
        {
          system: "http://namaste.who.int/CodeSystem/namaste",
          code: namasteEntry.NAMASTE_Code,
          display: `${namasteEntry.Traditional_Term} (${namasteEntry.English_Translation})`,
          version: "1.0"
        },
        // ICD-11 codings
        ...icd11Codes.map(icd => ({
          system: "http://id.who.int/icd/release/11/2022-02",
          code: icd.ICD11_Code,
          display: icd.Title,
          version: "2022-02"
        }))
      ],
      text: `${namasteEntry.Traditional_Term} - ${namasteEntry.English_Translation}`
    },
    subject: {
      reference: `Patient/${patientId}`
    },
    recordedDate: new Date().toISOString(),
    recorder: {
      display: clinician
    },
    extension: [
      {
        url: "http://namaste.who.int/StructureDefinition/medical-system",
        valueString: namasteEntry.Medical_System
      },
      {
        url: "http://namaste.who.int/StructureDefinition/mapping-info",
        extension: [
          {
            url: "mappingExists",
            valueBoolean: !!mapping
          },
          {
            url: "confidenceScore",
            valueDecimal: mapping ? parseFloat(mapping.Confidence_Score) : (mlSuggestion ? mlSuggestion.score : 0)
          },
          {
            url: "mappingType",
            valueString: mapping ? mapping.Mapping_Type : "ML_Predicted"
          }
        ]
      }
    ]
  };

  res.json({
    fhirCondition,
    mappingInfo: {
      existingMapping: !!mapping,
      mlSuggestion: mlSuggestion,
      icd11Codes: icd11Codes.length
    }
  });
});

// Proxy to ML service for suggestions
router.get('/ml/suggest', async (req, res) => {
  const { term } = req.query;
  
  if (!term) {
    return res.status(400).json({ 
      error: 'term parameter is required' 
    });
  }

  try {
    const response = await axios.get(`${ML_SERVICE_URL}/predict`, {
      params: { text: term },
      timeout: 10000
    });
    
    res.json({
      term,
      suggestions: response.data,
      source: 'ml-service'
    });
  } catch (error) {
    console.error('❌ ML service error:', error.message);
    
    // Fallback to simple text matching
    const fallbackResults = dataCache.icd11
      .filter(icd => icd.Title.toLowerCase().includes(term.toLowerCase()))
      .slice(0, 5)
      .map(icd => ({
        icd_code: icd.ICD11_Code,
        title: icd.Title,
        score: 0.5 // Fallback score
      }));
    
    res.json({
      term,
      suggestions: fallbackResults,
      source: 'fallback-search',
      note: 'ML service unavailable, using fallback text matching'
    });
  }
});

// Reload data endpoint (useful for development)
router.post('/reload-data', async (req, res) => {
  try {
    await initializeData();
    res.json({ 
      message: 'Data reloaded successfully',
      timestamp: new Date().toISOString(),
      counts: {
        namaste: dataCache.namaste.length,
        icd11: dataCache.icd11.length,
        mappings: dataCache.mappings.length
      }
    });
  } catch (error) {
    res.status(500).json({ 
      error: 'Failed to reload data',
      message: error.message 
    });
  }
});

module.exports = router;