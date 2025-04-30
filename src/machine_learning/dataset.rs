use ndarray::{Array1, Array2};
use std::error::Error;
use std::path::Path;
use std::fs::File;
use std::io::{BufRead, BufReader};

// Define a concrete error type
#[derive(Debug)]
pub enum DatasetError {
    IndexOutOfBounds(usize, usize),
    DimensionMismatch(String),
    IOError(std::io::Error),
    // Add other error variants as needed
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::IndexOutOfBounds(idx, len) =>
                write!(f, "Index {} out of bounds for dataset of length {}", idx, len),
            DatasetError::DimensionMismatch(msg) =>
                write!(f, "Dimension mismatch: {}", msg),
            DatasetError::IOError(err) =>
                write!(f, "IO error: {}", err),
        }
    }
}

impl std::error::Error for DatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DatasetError::IOError(err) => Some(err),
            _ => None,
        }
    }
}

/// Trait representing a dataset
pub trait Dataset {
    type Input;
    type Target;
    type Error: std::error::Error;

    /// Get the number of samples in the dataset
    fn len(&self) -> usize;

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a batch of samples
    fn get_batch(&self, indices: &[usize]) -> Result<(Vec<Self::Input>, Vec<Self::Target>), Self::Error>;

    /// Get a single sample
    fn get_sample(&self, index: usize) -> Result<(Self::Input, Self::Target), Self::Error>;
}

/// A simple dataset implementation for tabular data
pub struct TabularDataset {
    features: Array2<f64>,
    targets: Array2<f64>,
}

impl TabularDataset {
    /// Create a new dataset from feature and target arrays
    pub fn new(features: Array2<f64>, targets: Array2<f64>) -> Result<Self, Box<dyn Error>> {
        if features.nrows() != targets.nrows() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Number of feature rows ({}) does not match number of target rows ({})",
                    features.nrows(),
                    targets.nrows()
                ),
            )));
        }

        Ok(TabularDataset { features, targets })
    }

    /// Load a dataset from a CSV file
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        has_header: bool,
        feature_cols: &[usize],
        target_cols: &[usize],
        delimiter: char,
    ) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Skip header if needed
        if has_header {
            lines.next();
        }

        let mut features_data = Vec::new();
        let mut targets_data = Vec::new();

        for line in lines {
            let line = line?;
            let values: Vec<&str> = line.split(delimiter).collect();

            let mut feature_row = Vec::new();
            let mut target_row = Vec::new();

            for &col in feature_cols {
                if col < values.len() {
                    let value = values[col].parse::<f64>()?;
                    feature_row.push(value);
                } else {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Feature column index {} out of bounds", col),
                    )));
                }
            }

            for &col in target_cols {
                if col < values.len() {
                    let value = values[col].parse::<f64>()?;
                    target_row.push(value);
                } else {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Target column index {} out of bounds", col),
                    )));
                }
            }

            features_data.push(feature_row);
            targets_data.push(target_row);
        }

        // Convert to ndarray
        let n_samples = features_data.len();
        let n_features = if n_samples > 0 { features_data[0].len() } else { 0 };
        let n_targets = if n_samples > 0 { targets_data[0].len() } else { 0 };

        let mut features = Array2::zeros((n_samples, n_features));
        let mut targets = Array2::zeros((n_samples, n_targets));

        for i in 0..n_samples {
            for j in 0..n_features {
                features[[i, j]] = features_data[i][j];
            }

            for j in 0..n_targets {
                targets[[i, j]] = targets_data[i][j];
            }
        }

        Ok(TabularDataset { features, targets })
    }

    /// Normalize features using min-max scaling
    pub fn normalize_min_max(&mut self) {
        let n_features = self.features.ncols();

        for j in 0..n_features {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;

            // Find min and max
            for i in 0..self.features.nrows() {
                let val = self.features[[i, j]];
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }

            // Apply normalization
            let range = max_val - min_val;
            if range > 1e-10 {
                for i in 0..self.features.nrows() {
                    self.features[[i, j]] = (self.features[[i, j]] - min_val) / range;
                }
            }
        }
    }

    /// Normalize features using z-score (mean=0, std=1)
    pub fn normalize_z_score(&mut self) {
        let n_samples = self.features.nrows();
        let n_features = self.features.ncols();

        for j in 0..n_features {
            // Calculate mean
            let mut mean = 0.0;
            for i in 0..n_samples {
                mean += self.features[[i, j]];
            }
            mean /= n_samples as f64;

            // Calculate standard deviation
            let mut std_dev = 0.0;
            for i in 0..n_samples {
                let diff = self.features[[i, j]] - mean;
                std_dev += diff * diff;
            }
            std_dev = (std_dev / n_samples as f64).sqrt();

            // Apply normalization
            if std_dev > 1e-10 {
                for i in 0..n_samples {
                    self.features[[i, j]] = (self.features[[i, j]] - mean) / std_dev;
                }
            }
        }
    }
}

impl Dataset for TabularDataset {
    type Input = Array1<f64>;
    type Target = Array1<f64>;
    type Error = DatasetError;

    fn len(&self) -> usize {
        self.features.nrows()
    }

    fn get_batch(&self, indices: &[usize]) -> Result<(Vec<Self::Input>, Vec<Self::Target>), Self::Error> {
        let mut batch_features = Vec::with_capacity(indices.len());
        let mut batch_targets = Vec::with_capacity(indices.len());

        for &idx in indices {
            if idx >= self.len() {
                return Err(DatasetError::IndexOutOfBounds(idx, self.len()));
            }

            let feature_row = self.features.row(idx).to_owned();
            let target_row = self.targets.row(idx).to_owned();

            batch_features.push(feature_row);
            batch_targets.push(target_row);
        }

        Ok((batch_features, batch_targets))
    }

    fn get_sample(&self, index: usize) -> Result<(Self::Input, Self::Target), Self::Error> {
        if index >= self.len() {
            return Err(DatasetError::IndexOutOfBounds(index, self.len()));
        }

        let feature_row = self.features.row(index).to_owned();
        let target_row = self.targets.row(index).to_owned();

        Ok((feature_row, target_row))
    }
}

/// Utility for splitting a dataset into training and testing sets
pub struct TrainTestSplit<T: Dataset> {
    dataset: T,
    train_indices: Vec<usize>,
    test_indices: Vec<usize>,
}

impl<T: Dataset> TrainTestSplit<T> {
    /// Create a new train-test split
    pub fn new(dataset: T, test_ratio: f64, shuffle: bool) -> Self {
        let n_samples = dataset.len();
        let n_test = (n_samples as f64 * test_ratio).round() as usize;
        let n_train = n_samples - n_test;

        let mut indices: Vec<usize> = (0..n_samples).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        let train_indices = indices[0..n_train].to_vec();
        let test_indices = indices[n_train..].to_vec();

        TrainTestSplit {
            dataset,
            train_indices,
            test_indices,
        }
    }

    /// Get the training set
    pub fn get_train_set(&self) -> Result<(Vec<T::Input>, Vec<T::Target>), T::Error> {
        self.dataset.get_batch(&self.train_indices)
    }

    /// Get the test set
    pub fn get_test_set(&self) -> Result<(Vec<T::Input>, Vec<T::Target>), T::Error> {
        self.dataset.get_batch(&self.test_indices)
    }

    /// Get the number of training samples
    pub fn train_len(&self) -> usize {
        self.train_indices.len()
    }

    /// Get the number of test samples
    pub fn test_len(&self) -> usize {
        self.test_indices.len()
    }

    /// Get the underlying dataset
    pub fn dataset(&self) -> &T {
        &self.dataset
    }
}

/// Utility for iterating through a dataset in batches
pub struct DataLoader<T: Dataset> {
    dataset: T,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_index: usize,
}

impl<T: Dataset> DataLoader<T> {
    /// Create a new data loader
    pub fn new(dataset: T, batch_size: usize, shuffle: bool) -> Self {
        let n_samples = dataset.len();
        let indices: Vec<usize> = (0..n_samples).collect();

        DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_index: 0,
        }
    }

    /// Reset the data loader for a new epoch
    pub fn reset(&mut self) {
        self.current_index = 0;

        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get the next batch
    pub fn next_batch(&mut self) -> Option<Result<(Vec<T::Input>, Vec<T::Target>), T::Error>> {
        if self.current_index >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_index + self.batch_size).min(self.dataset.len());
        let batch_indices = self.indices[self.current_index..end_idx].to_vec();

        self.current_index = end_idx;

        Some(self.dataset.get_batch(&batch_indices))
    }
}
