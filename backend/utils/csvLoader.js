const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

class CSVLoader {
  constructor() {
    this.datasetsPath = path.join(__dirname, '../../datasets');
  }

  /**
   * Generic CSV loader with error handling and validation
   */
  async loadCSV(filename, expectedHeaders = []) {
    const filePath = path.join(this.datasetsPath, filename);
    const results = [];

    return new Promise((resolve, reject) => {
      // Check if file exists
      if (!fs.existsSync(filePath)) {
        console.warn(`⚠️  CSV file not found: ${filename}`);
        console.warn(`Expected path: ${filePath}`);
        resolve([]);
        return;
      }

      let headerValidated = false;
      let rowCount = 0;
      let errorCount = 0;

      fs.createReadStream(filePath)
        .pipe(csv())
        .on('headers', (headers) => {
          // Validate headers
          if (expectedHeaders.length > 0) {
            const missingHeaders = expectedHeaders.filter(h => !headers.includes(h));
            if (missingHeaders.length > 0) {
              console.warn(`⚠️  Missing headers in ${filename}: ${missingHeaders.join(', ')}`);
            }
          }
          headerValidated = true;
          console.log(`📋 Loading ${filename} with headers: ${headers.join(', ')}`);
        })
        .on('data', (data) => {
          try {
            // Skip rows with missing required fields
            if (expectedHeaders.length > 0) {
              const hasAllRequiredFields = expectedHeaders.every(header => 
                data[header] !== undefined && data[header].toString().trim() !== ''
              );
              
              if (!hasAllRequiredFields) {
                console.warn(`⚠️  Skipping incomplete row in ${filename}:`, data);
                errorCount++;
                return;
              }
            }

            results.push(data);
            rowCount++;
          } catch (error) {
            console.error(`❌ Error parsing row in ${filename}:`, error);
            errorCount++;
          }
        })
        .on('end', () => {
          console.log(`✅ Successfully loaded ${filename}: ${rowCount} rows, ${errorCount} errors`);
          resolve(results);
        })
        .on('error', (error) => {
          console.error(`❌ Error reading ${filename}:`, error);
          resolve([]); // Return empty array instead of rejecting
        });
    });
  }

  /**
   * Load NAMASTE dataset
   */
  async loadNamaste() {
    const expectedHeaders = ['NAMASTE_Code', 'Traditional_Term', 'English_Translation', 'Medical_System', 'Category'];
    return await this.loadCSV('namaste_100_dataset.csv', expectedHeaders);
  }

  /**
   * Load ICD-11 dataset
   */
  async loadIcd11() {
    const expectedHeaders = ['ICD11_Code', 'Title', 'Module', 'Category', 'Code_Type'];
    return await this.loadCSV('icd11_100_dataset.csv', expectedHeaders);
  }

  /**
   * Load mapping dataset
   */
  async loadMappings() {
    const expectedHeaders = ['NAMASTE_Code', 'ICD11_TM2_Code', 'ICD11_Biomedicine_Code', 'Mapping_Type', 'Confidence_Score'];
    return await this.loadCSV('namaste_icd11_100_mapping.csv', expectedHeaders);
  }

  /**
   * Load all datasets at once
   */
  async loadAllData() {
    console.log('🔄 Loading all CSV datasets...');
    
    const [namaste, icd11, mappings] = await Promise.all([
      this.loadNamaste(),
      this.loadIcd11(),
      this.loadMappings()
    ]);

    console.log(`📊 Data loading complete:`);
    console.log(`  • NAMASTE entries: ${namaste.length}`);
    console.log(`  • ICD-11 entries: ${icd11.length}`);
    console.log(`  • Mapping entries: ${mappings.length}`);

    return { namaste, icd11, mappings };
  }
}

module.exports = new CSVLoader();