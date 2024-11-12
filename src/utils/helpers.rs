use opencv::imgproc::INTER_LINEAR;
use opencv::objdetect::CascadeClassifier;
use opencv::{core, imgproc, prelude::*};
use std::collections::{HashSet, VecDeque};

use super::constants::K_FAST_EYE_WIDTH;
use super::constants::K_WEIGHT_DIVISOR;

pub fn load_detector(detector_path: &str) -> opencv::Result<CascadeClassifier> {
    Ok(CascadeClassifier::new(detector_path)?)
}

pub fn compute_roi(mat: &Mat, sub_mat: &core::Rect_<i32>) -> opencv::Result<Mat> {
    mat.roi(*sub_mat)?;
    Ok(mat.roi(*sub_mat)?.clone_pointee())
}

pub fn scale_to_fast_size(src: &Mat, dst: &mut Mat) {
    imgproc::resize(
        src,
        dst,
        core::Size::new(
            K_FAST_EYE_WIDTH,
            ((K_FAST_EYE_WIDTH as f64 / src.cols() as f64) * src.rows() as f64) as i32,
        ),
        0.,
        0.,
        INTER_LINEAR,
    );
}

pub fn compute_gradient(mat: &Mat) -> opencv::Result<Mat> {
    let mut out = Mat::zeros(mat.rows(), mat.cols(), core::CV_64F)?.to_mat()?;
    for y in 0..mat.rows() {
        for x in 0..mat.cols() {
            let gradient = if x == 0 {
                let val_1 = *mat.at_2d::<u8>(y, x + 1)? as f64;
                let val_2 = *mat.at_2d::<u8>(y, x)? as f64;
                val_1 - val_2
            } else if x < mat.cols() - 1 {
                let val_1 = *mat.at_2d::<u8>(y, x + 1)? as f64;
                let val_2 = *mat.at_2d::<u8>(y, x - 1)? as f64;
                (val_1 - val_2) / 2.0
            } else {
                let val_1 = *mat.at_2d::<u8>(y, x)? as f64;
                let val_2 = *mat.at_2d::<u8>(y, x - 1)? as f64;
                val_1 - val_2
            };
            *out.at_2d_mut::<f64>(y, x)? = gradient;
        }
    }

    Ok(out)
}

pub fn compute_magnitudes(x_grad: &Mat, y_grad: &Mat) -> opencv::Result<Mat> {
    let mut magnitude_mat: Mat =
        Mat::zeros(x_grad.rows(), x_grad.cols(), core::CV_64F)?.to_mat()?;
    for row in 0..x_grad.rows() {
        for col in 0..x_grad.cols() {
            let grad_x_value = *x_grad.at_2d::<f64>(row, col)?;
            let grad_y_value = *y_grad.at_2d::<f64>(row, col)?;
            let magnitude = ((grad_x_value * grad_x_value) + (grad_y_value * grad_y_value)).sqrt();
            *magnitude_mat.at_2d_mut::<f64>(row, col)? = magnitude;
        }
    }
    Ok(magnitude_mat)
}

pub fn compute_dynamic_treshold(mat: &Mat, treshold: f64) -> opencv::Result<f64> {
    let mut std_magn_grad = core::Scalar::all(0.0);
    let mut mean_magn_grad = core::Scalar::all(0.0);

    core::mean_std_dev(
        &mat,
        &mut mean_magn_grad,
        &mut std_magn_grad,
        &core::no_array(),
    )?;
    let std_dev = std_magn_grad[0] / (mat.rows() as f64 * mat.cols() as f64).sqrt();
    Ok(treshold * std_dev + mean_magn_grad[0])
}

pub fn test_possibile_centers_formula(
    x: i32,
    y: i32,
    weight_mat: &Mat,
    out: &mut Mat,
    gx: f64,
    gy: f64,
) -> opencv::Result<()> {
    for cy in 0..out.rows() {
        for cx in 0..out.cols() {
            if x == cx && y == cy {
                continue;
            }
            let mut dx = (x - cx) as f64;
            let mut dy = (y - cy) as f64;

            let magnitude = ((dx * dx) + (dy * dy)).sqrt();
            dx = dx / magnitude;
            dy = dy / magnitude;

            let mut dot_product = dx * gx + dy * gy;
            dot_product = dot_product.max(0.);
            let weight = *weight_mat.at_2d::<u8>(cy, cx)?;
            *out.at_2d_mut::<f64>(cy, cx)? +=
                dot_product.powi(2) * (weight as f64 / K_WEIGHT_DIVISOR);
        }
    }

    Ok(())
}

pub fn flood_kill_edges(mat: &mut Mat) -> opencv::Result<Mat> {
    // Initialize the border
    imgproc::rectangle(
        mat,
        core::Rect::new(0, 0, mat.cols(), mat.rows()),
        core::Scalar::new(255.0, 0.0, 0.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    let mut mask = Mat::new_rows_cols_with_default(
        mat.rows(),
        mat.cols(),
        core::CV_8U,
        core::Scalar::all(255.),
    )?;

    let mut to_do = VecDeque::new();
    let mut visited = HashSet::new(); // Track visited points
    to_do.push_front(core::Point::new(0, 0));
    visited.insert((0, 0)); // Mark starting point as visited

    while let Some(p) = to_do.pop_front() {
        if *mat.at_2d::<f32>(p.y, p.x)? == 0f32 {
            continue;
        }

        *mat.at_2d_mut::<f32>(p.y, p.x)? = 0f32;
        *mask.at_2d_mut::<u8>(p.y, p.x)? = 0u8;

        let neighbors = [
            core::Point::new(p.x + 1, p.y), // right
            core::Point::new(p.x - 1, p.y), // left
            core::Point::new(p.x, p.y + 1), // down
            core::Point::new(p.x, p.y - 1), // up
        ];

        for np in neighbors {
            let point_tuple = (np.x, np.y);

            // Only process point if:
            // 1. It's within matrix bounds
            // 2. It hasn't been visited before
            // 3. It's not black (0)
            if in_mat(np, &mat)?
                && !visited.contains(&point_tuple)
                && *mat.at_2d::<f32>(np.y, np.x)? != 0f32
            {
                to_do.push_back(np);
                visited.insert(point_tuple);
            }
        }
    }

    Ok(mask)
}

fn in_mat(p: core::Point, mat: &Mat) -> opencv::Result<bool> {
    Ok(p.x >= 0 && p.y >= 0 && p.x < mat.cols() && p.y < mat.rows())
}

pub fn unscale_point(p: core::Point, orig_size: &core::Rect) -> opencv::Result<core::Point> {
    let ratio = K_FAST_EYE_WIDTH as f64 / orig_size.width as f64;
    let x = p.x as f64 / ratio;
    let y = p.y as f64 / ratio;
    Ok(core::Point::new(x as i32, y as i32))
}
