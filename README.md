# Technical Report: Kharif 2026 Crop Classification & Agricultural Intelligence MVP
Study Area: Chitradurga District, Karnataka, India
Target Window: June 1, 2026 – October 30, 2026 (Kharif Season)

1. Executive Summary & Pipeline Architecture
This operational remote sensing pipeline delivers wall-to-wall crop mapping, spatial uncertainty quantification, and crop area statistics for Chitradurga District during the Kharif 2026 season. Chitradurga lies within Karnataka’s Central Dry Agro-Climatic Zone (Zone-4), characterized by semi-arid red sandy loams, shallow black soils, and high inter-annual monsoon variability.
The workflow integrates multi-temporal Sentinel-2 optical observations (restricted to NDVI) and Sentinel-1 (SAR) backscatter. Ingestion, cloud masking via Cloud Score+, temporal compositing, and sampling occur on Google Earth Engine (GEE). Feature gap-filling via Savitzky-Golay filtering, spatial machine learning benchmarking, block-wise raster inference, and final cartographic rendering are executed locally in Python.
<img width="515" height="560" alt="Screenshot 2026-11-08 023436" src="https://github.com/user-attachments/assets/b23dde44-5328-471c-8cdb-d84273b20127" />

## 2. Key Decisions & Technical Trade-Offs
### 2.0 Data Improvisation:
The GT data provided was made of lat/long points instead of polygons. Polygons were made by hand for each GT data point with rough estimates of plot boundary by making the extent of the plot conservatively, so as not to extend beyond the actual plot limit. Buffer of the point was not done as some points were very close to the plot boundary near the road.
### 2.1 Optical (NDVI Only) + SAR Integration vs. Optical Alone
The Indian Southwest Monsoon creates heavy cloud cover across Chitradurga between late June and late August. Optical sensors alone experience contiguous data dropouts during critical germination and early tillering/canopy development.
Sentinel-1 penetrates convective cloud layers, providing continuous structural measurements. Cross-polarization (VH) directly responds to volume scattering from crop canopy architecture, while the VH/VV cross-ratio normalizes environmental soil moisture spikes following rainfall events.
Restricting Sentinel-2 to NDVI minimizes high-dimensional collinearity, avoids inter-band atmospheric scattering biases in the blue/green spectrum, and reduces memory consumption during raster tiling.
### 2.2 Cloud Masking: Cloud Score+ 
Integrated Cloud Score+ Harmonized (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED), applying a strict threshold on the clear-sky quality band:  cs≥0.60
This removes semi-transparent cirrus and localized cloud edges at the native pixel level before computing NDVI.
### 2.3 Temporal Window: 12-Day Moving Windows vs. Monthly Composites
Monthly composites flatten dynamic phenological transitions. For example, maize transitions from vegetative emergence to tasseling within 25 days.
We implemented 12-day moving windows centered on target step dates (t ±6 days) across June 1 to November 30, 2026. This yielded 16 distinct time intervals, aligning with the 5-day revisit rate of the Sentinel-2 constellation while capturing distinct vegetative inflection points.
### 2.4 Cropland Masking: Dynamic World 2026 vs. Static Baselines (ESA WorldCover)
Static baselines (such as ESA WorldCover 2021) do not capture Kharif 2026 fallow rotations, rainfed abandonment, or recent horticultural expansion.
We queried Google Dynamic World (GOOGLE/DYNAMICWORLD/V1) filtered strictly to the Kharif 2026 operational window:
P(Active Vegetation)=P(crops)≥0.25
This masks non-vegetated terrain (urban settlements, granite hillocks, solar parks, water bodies) in real time before model inference.
### 2.5 Gap-Filling: Savitzky-Golay Filtering vs. Linear Interpolation
Linear interpolation produces flat ramps across missing data intervals, blunting phenological peaks.
We applied a Savitzky-Golay polynomial filter (scipy.signal.savgol_filter, window length = 5, polynomial order = 2) across the 16-point NDVI vector.
Strict Mask Rule: If a pixel had no valid optical observations across the entire 16-window sequence, it was not interpolated. Hallucinating vegetative profiles on permanently clouded or unobserved ground introduces severe classification errors.
### 2.6 ML Architecture: Gradient Boosted Trees (LightGBM) vs Random Forest Classifier 
LightGBM provides superior convergence speed, handles missing values natively, captures non-linear split dynamics across SAR and optical indices, and calculates class prediction probabilities needed for entropy modeling.
Random Forest gave slightly worse results than LightGBM.
### 2.7 Spatial Validation: Spatial Group Block CV vs. Random K-Fold
Standard random k-fold cross-validation results in data leakage because adjacent pixels from the same agricultural field share identical soils, sowing dates, and management practices.
We implemented Stratified Group 5-Fold Cross-Validation (StratifiedGroupKFold) grouped by unique Polygon / Field ID. Pixels from the same physical field were never split across training and validation partitions.

## 3. Ground Truth Scope, Class Profiling & Spatial Validation

### 3.1 Ground Truth Ingestion & Dynamic Class Thresholding
To prevent the classifier from fitting unrepresentative decision boundaries on scarce classes, an empirical rule was applied: Classes supported by >25 verified field polygons were retained as individual target classes; all remaining classes were aggregated into 'Others'.
This established 4 target classes:
Arecanut: Perennial horticultural plantation; high, stable biomass year-round; localized in canal/borewell-irrigated belts (Hosadurga, Holalkere).
Maize: High-biomass cereal; rapid vegetative surge through July/August; high peak NDVI (>0.80); extensive across rainfed taluks.
Ragi: Low-to-moderate biomass staple cereal; later vegetative peak than maize; lower NIR reflectance.
Others: Encompasses minority crops (e.g., smallholder pulses, vegetables, minor millets).
<img width="590" height="277" alt="Screenshot 2026-11-08 022719" src="https://github.com/user-attachments/assets/77acb4f8-91a2-4146-8b92-a308f02961bf" />

### 3.2 Spatial Group Cross-Validation Performance (LightGBM)
## insert cross val image and confusion matrix image on github
<img width="500" height="411" alt="Screenshot 2026-11-07 230724" src="https://github.com/user-attachments/assets/f3bc7d2a-c748-447d-aab8-bc345d02d699" />
<img width="500" height="411" alt="Screenshot 2026-11-08 024104" src="https://github.com/user-attachments/assets/8272039c-4cb9-42ca-8401-8dcea9a15c55" />

#### Confusion Dynamics & Agronomic Evaluation:
Arecanut Distinctions: Exhibits an elevated baseline NDVI (>0.60) and high VH backscatter early in the season (June–July) when field crops are just emerging from bare soil.

Maize vs. Ragi: Represents the primary field-crop confusion. Maize establishes peak leaf area index (LAI) 15–20 days faster than Ragi, producing an earlier peak in the 12-day NDVI series and higher radar volume scattering in SAR VH.

Below images represent the raw(without cloud mask) pixel timeseries of Maize and Ragi crop. We should note that both the signals starts rising after 9th August, Ragi signal rises a little slower than Maize. But still both the signals are quite identical and hence the classification confusion.
<img width="1300" height="750" alt="pixel-wise-ndvi-time-series-ragi" src="https://github.com/user-attachments/assets/4eba3c76-e7e4-44c7-b1dd-58e2163c4ab0" />
<img width="1300" height="750" alt="pixel-wise-ndvi-time-series-maize" src="https://github.com/user-attachments/assets/cf1b4076-bb05-48b8-b7be-b239928af791" />


Others Classification: Shows lower precision (0.450) due to wide intra-class spectral variance.
## 4. District Area Statistics & Uncertainty Quantification
### 4.1 Ingestion from area_stats.csv
Model predictions were generated using block-wise parallel inference on the Chitradurga raster stack, masked by Dynamic World 2026 active vegetation. Class-wise acreages were compiled directly into area_stats.csv.
### 4.2 Prediction Uncertainty: Normalized Shannon Entropy
To move beyond single-label deterministic assumptions, prediction confidence was calculated for every pixel using Normalized Shannon Entropy.
## 5. Lightweight MVP Implementation (Folium Architecture)
The MVP delivery requirement was met by generating a standalone, interactive HTML map (chitradurga_crop_map_2026.html) using Python's folium library.
Code
chitradurga_crop_map_2026.html
├── Leaflet Map Canvas (Google Satellite Hybrid & CartoDB Positron Basemaps)
├── Classification Layer (Dynamic RGBA PNG: Arecanut, Maize, Ragi, Others)
├── Uncertainty Heatmap (Dynamic RGBA PNG: Normalized Shannon Entropy RdYlGn_r)
├── Layer Control (Top-Right Toggle for Real-Time Inspection)
└── Floating Dashboard Panel (Bottom-Left)
    ├── Thematic Class Color Swatches
    ├── Entropy Scale Gradient Bar (0.0 Certain -> 1.0 Ambiguous)
    └── Dynamic HTML Table (Directly bound to area_stats.csv)
Key Technical Aspects of the MVP:
Memory-Safe RGBA Raster Export: The classification and entropy GeoTIFFs are converted into web-optimized, georeferenced RGBA PNG overlays. Non-agricultural pixels are assigned an alpha value of 0 (100% transparent), allowing underlying satellite basemaps to remain visible.
Dynamic CSV Data Binding: The dashboard panel parses area_stats.csv directly at generation time, formatting and rendering area metrics without hardcoded values.
Zero Server Dependency: The output is completely self-contained, requiring no running Python server, Streamlit instance, or backend database, enabling rapid distribution to field stakeholders.
## 6. Risks, Portability & Production Scaling Roadmap
### 6.1 Portability Risks & Model Limitations
Temporal Transferability: The decision boundaries learned for Kharif 2026 cannot be applied directly to Rabi or Summer seasons due to changes in crop mixes (e.g., Bengal Gram, Sunflower), and altered phenology.
Geographic Portability: Applying this model to a very different region might not work as the seasonality(duration) of same crops can change, and the list of crops might change too.
### 6.2 Two-Week Production Expansion Roadmap
If granted additional Time and Support:
Extensive Data Collection:
With more data samples, both for target and 'other' crops(equally important), and accurate sowing and harvesting dates, the model should perform significantly better. Data points should have accurate polygon boundaries.
Explore Other Data Sources:
Explore data sources with better temporal and spatial resolution. 
Try giving more importance to SAR based features as Kharif season affects optical data very harshly.
## Acknowledgement:
Google AI Studio was used to generate significant parts of the code in pieces and finally put together to make the pipeline complete. AI was also used to help complete the write-up.
