//! Progress reporting for long-running algorithms.
//!
//! This module provides a simple progress callback mechanism that algorithms
//! can use to report their progress to callers.
//!
//! # Example
//!
//! ```ignore
//! use morsel::algo::progress::Progress;
//!
//! let progress = Progress::new(|current, total, message| {
//!     println!("[{}/{}] {}", current, total, message);
//! });
//!
//! // Pass to algorithm options
//! let options = SmoothOptions::default().with_progress(progress);
//! ```

/// A progress callback that receives updates during long-running operations.
///
/// The callback receives:
/// - `current`: Current step (0-based)
/// - `total`: Total number of steps
/// - `message`: Description of the current operation
pub struct Progress {
    callback: Box<dyn Fn(usize, usize, &str) + Send + Sync>,
}

impl Progress {
    /// Create a new progress reporter with the given callback.
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(usize, usize, &str) + Send + Sync + 'static,
    {
        Self {
            callback: Box::new(callback),
        }
    }

    /// Report progress.
    #[inline]
    pub fn report(&self, current: usize, total: usize, message: &str) {
        (self.callback)(current, total, message);
    }

    /// Report progress within a sub-range.
    ///
    /// Maps progress from `[0, sub_total]` to `[range_current, range_current + 1]`
    /// within a total of `range_total` steps. This enables hierarchical progress
    /// where sub-operations report their progress within an allocated slice.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Top level has 4 steps. Step 0 is "split edges" which itself has many sub-steps.
    /// // Report sub-progress within step 0:
    /// progress.report_sub(edges_done, total_edges, 0, 4, "Splitting edges");
    /// ```
    #[inline]
    pub fn report_sub(
        &self,
        sub_current: usize,
        sub_total: usize,
        range_current: usize,
        range_total: usize,
        message: &str,
    ) {
        if sub_total == 0 || range_total == 0 {
            return;
        }
        // Map sub-progress to the range [range_current, range_current + 1)
        // Using fixed-point math to avoid floating point: multiply by 1000 for precision
        let sub_fraction = (sub_current * 1000) / sub_total;
        let effective = range_current * 1000 + sub_fraction;
        let total_scaled = range_total * 1000;
        (self.callback)(effective, total_scaled, message);
    }

    /// Create a no-op progress reporter that discards all updates.
    pub fn none() -> Self {
        Self::new(|_, _, _| {})
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self::none()
    }
}

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Progress").finish_non_exhaustive()
    }
}
