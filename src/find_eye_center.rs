use crate::utils::helpers::{
    compute_dynamic_treshold, compute_gradient, compute_magnitudes, compute_roi,
    scale_to_fast_size, unscale_point,
};
use crate::utils::{constants, helpers};

use opencv::core::CV_32F;
use opencv::{
    core,
    imgproc::{self, LINE_8},
    prelude::*,
};

pub fn find_eye_center(gray_frame: &Mat, eye: &core::Rect) -> opencv::Result<core::Point> {
    let eye_roi_unscaled = compute_roi(&gray_frame, &eye)?;
    let mut eye_roi = Mat::default();
    scale_to_fast_size(&eye_roi_unscaled, &mut eye_roi);

    //-- Find the gradient
    let mut x_grad = helpers::compute_gradient(&eye_roi)?;
    let mut eye_roi_transposed = Mat::default();
    core::transpose(&eye_roi, &mut eye_roi_transposed)?;
    let mut y_grad = Mat::default();
    core::transpose(
        &helpers::compute_gradient(&eye_roi_transposed)?,
        &mut y_grad,
    )?;

    // Compute gradient's magnitudes
    let gradients_mat = compute_magnitudes(&x_grad, &y_grad)?;

    // Gradient treshold
    let grad_treshold = compute_dynamic_treshold(&gradients_mat, constants::K_GRADIENT_TRESHOLD)?;

    for row in 0..eye_roi.rows() {
        for col in 0..eye_roi.cols() {
            let magnitude = *gradients_mat.at_2d::<f64>(row, col)?;
            if magnitude > grad_treshold {
                let x_val = *x_grad.at_2d::<f64>(row, col)?;
                let y_val = *y_grad.at_2d::<f64>(row, col)?;
                *x_grad.at_2d_mut::<f64>(row, col)? = x_val / magnitude;
                *y_grad.at_2d_mut::<f64>(row, col)? = y_val / magnitude;
            } else {
                *x_grad.at_2d_mut::<f64>(row, col)? = 0.0;
                *y_grad.at_2d_mut::<f64>(row, col)? = 0.0;
            }
        }
    }

    let mut weight = Mat::default();
    imgproc::gaussian_blur(
        &eye_roi,
        &mut weight,
        core::Size::new(constants::K_WEIGHT_BLUR_SIZE, constants::K_WEIGHT_BLUR_SIZE),
        0.,
        0.,
        core::BORDER_DEFAULT,
    )?;

    for row in 0..weight.rows() {
        for col in 0..weight.cols() {
            let value = *weight.at_2d::<u8>(row, col)?;
            *weight.at_2d_mut::<u8>(row, col)? = 255 - value;
        }
    }

    let mut out_sum = Mat::zeros(eye_roi.rows(), eye_roi.cols(), core::CV_64F)?.to_mat()?;

    for row in 0..weight.rows() {
        for col in 0..weight.cols() {
            let grad_x_val = *x_grad.at_2d::<f64>(row, col)?;
            let grad_y_val = *y_grad.at_2d::<f64>(row, col)?;
            if grad_x_val == 0. && grad_y_val == 0. {
                continue;
            }

            helpers::test_possibile_centers_formula(
                col,
                row,
                &weight,
                &mut out_sum,
                grad_x_val,
                grad_y_val,
            )?;
        }
    }

    let num_gradients: f64 = (weight.rows() * weight.cols()) as f64;
    let mut out = Mat::default();
    out_sum.convert_to(&mut out, CV_32F, 1.0 / num_gradients, 0.);

    let mut max_val = 0f64;
    let mut max_loc = core::Point::new(0, 0);
    core::min_max_loc(
        &out,
        None,
        Some(&mut max_val),
        None,
        Some(&mut max_loc),
        &core::no_array(),
    );

    let mut flood_clone = Mat::default();
    imgproc::threshold(
        &out,
        &mut flood_clone,
        max_val * constants::K_POST_PROCESS_THRESHOLD,
        0.,
        imgproc::THRESH_TOZERO,
    );

    let mask = helpers::flood_kill_edges(&mut flood_clone)?;
    let mut display_mask = Mat::default();
    core::normalize(
        &mask,
        &mut display_mask,
        0.,
        255.,
        core::NORM_MINMAX,
        core::CV_8U,
        &core::no_array(),
    )?;

    core::min_max_loc(
        &out,
        None,
        Some(&mut max_val),
        None,
        Some(&mut max_loc),
        &mask,
    );

    Ok(unscale_point(max_loc, &eye)?)
}
