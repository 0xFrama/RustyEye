use opencv::{
    core::{self, Point, Rect_, Scalar, Vector},
    highgui, imgcodecs, imgproc,
    objdetect::{self, CascadeClassifier},
    prelude::*,
    types::{self, VectorOfRect},
};


use std::collections::VecDeque;

const PERC_WIDTH_EYE: i32 = 30;
const PERC_HEIGHT_EYE: i32 = 17;
const PERC_EYE_X: i32 = 13;
const PERC_EYE_Y: i32 = 23;
const EYE_WIDTH: i32 = 60;
const K_ENABLE_WEIGHT: bool = true;
const K_WEIGHT_DIVISOR: f64 = 1.0;
const GRADIENT_TRESHOLD: f64 = 50.0;
const POST_PROCESS_THRESHOLD: f64 = 90.;

fn convert_to_gray(src: &Mat) -> opencv::Result<Mat> {
    let mut gray_img = Mat::default();
    let mut gray_img_64f = Mat::default();

    imgproc::cvt_color(&src, &mut gray_img, imgproc::COLOR_BGR2GRAY, 1)?;
    gray_img.convert_to(&mut gray_img_64f, opencv::core::CV_64F, 1., 0.)?;

    Ok(gray_img_64f)
}

fn load_detector(detector_path: &str) -> opencv::Result<CascadeClassifier> {
    Ok(objdetect::CascadeClassifier::new(detector_path)?)
}

fn compute_roi(mat: &Mat, eye: &Rect_<i32>) -> opencv::Result<Mat> {
    Ok(mat.roi(*eye)?.clone_pointee())
}

fn compute_gradient(mat: &Mat) -> opencv::Result<Mat> {
    let mut out = Mat::zeros(mat.rows(), mat.cols(), core::CV_64F)?.to_mat()?;

    for y in 0..mat.rows() {
        for x in 0..mat.cols() {
            let gradient = if x == 0 {
                let val_1 = *mat.at_2d::<f64>(y, x + 1)?;
                let val_2 = *mat.at_2d::<f64>(y, x)?;
                val_1 - val_2
            } else if x < mat.cols() - 1 {
                let val_1 = *mat.at_2d::<f64>(y, x + 1)?;
                let val_2 = *mat.at_2d::<f64>(y, x - 1)?;
                (val_1 - val_2) / 2.0
            } else {
                let val_1 = *mat.at_2d::<f64>(y, x)?;
                let val_2 = *mat.at_2d::<f64>(y, x - 1)?;
                val_1 - val_2
            };

            *out.at_2d_mut::<f64>(y, x)? = gradient;
        }
    }

    Ok(out)
}

fn compute_magnitude(x_grad: &Mat, y_grad: &Mat) -> opencv::Result<Mat> {
    let mut magnitude_mat: Mat =
        Mat::zeros(x_grad.rows(), x_grad.cols(), core::CV_64F)?.to_mat()?;
    core::magnitude(x_grad, y_grad, &mut magnitude_mat)?;
    Ok(magnitude_mat)
}

fn compute_dynamic_treshold(mat: &Mat, treshold: f64) -> opencv::Result<f64> {
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

fn unscale_point(p: core::Point, orig: &Mat) -> opencv::Result<(i32, i32)> {
    // Calculate the ratio
    let ratio = EYE_WIDTH as f64 / orig.cols() as f64;

    // Calculate unscaled coordinates
    let x = (p.x as f64 / ratio).round() as i32;
    let y = (p.y as f64 / ratio).round() as i32;

    Ok((x, y))
}

fn original_size(p: core::Point, orig: &Rect_<i32>) -> opencv::Result<(i32, i32)> {
    Ok((p.x + orig.x, p.y + orig.y))
}

fn test_possibile_centers_formula(
    cx: i32,
    cy: i32,
    weight_mat: &Mat,
    out: &mut Mat,
    x_grad: &Mat,
    y_grad: &Mat,
) -> opencv::Result<()> {
    let mut dot_product_sum = 0.;
    for row in 0..out.rows() {
        for col in 0..out.cols() {
            let mut dx = (row - cx) as f64;
            let mut dy = (col - cy) as f64;

            let magnitude = ((dx * dx) + (dy * dy)).sqrt();
            dx = dx / magnitude;
            dy = dy / magnitude;

            let gx = *x_grad.at_2d::<f64>(row, col)?;
            let gy = *y_grad.at_2d::<f64>(row, col)?;

            if gx == 0. && gy == 0. {
                continue;
            }

            let mut dot_product = dx * gx + dy * gy;
            dot_product = dot_product.max(0.);
            let weight = *weight_mat.at_2d::<f64>(cx, cy)?;
            *out.at_2d_mut::<f64>(cx, cy)? = dot_product.powi(2) * (weight / K_WEIGHT_DIVISOR);
            dot_product_sum += weight * dot_product.powi(2);
        }
    }
    *out.at_2d_mut::<f64>(cx, cy)? = dot_product_sum;

    Ok(())
}

fn scale_img(src: &Mat) -> opencv::Result<Mat> {
    let mut dst = Mat::default();
    imgproc::resize(
        src,
        &mut dst,
        core::Size::new(EYE_WIDTH, (EYE_WIDTH * src.rows()) / src.cols() as i32),
        0.,
        0.,
        0,
    )?;

    Ok(dst)
}

fn create_binary_mask(mat: &Mat, threshold: f64) -> opencv::Result<Mat> {
    let mut binary_mask = Mat::new_rows_cols_with_default(
        mat.rows(),
        mat.cols(),
        core::CV_64F,
        core::Scalar::all(0.),
    )?;
    for row in 0..mat.rows() {
        for col in 0..mat.cols() {
            let grad = *mat.at_2d::<f64>(row, col)?;
            if grad > threshold {
                *binary_mask.at_2d_mut::<f64>(row, col)? = 1.;
            } else {
                *binary_mask.at_2d_mut::<f64>(row, col)? = 0.;
            }
        }
    }
    Ok(binary_mask)
}

fn is_valid(row: i32, col: i32, mat: &Mat) -> opencv::Result<bool> {
    if row < 0
        || row > mat.rows()
        || col < 0
        || col > mat.cols()
        || *mat.at_2d::<f64>(row, col)? == 0.
    {
        return Ok(false);
    }
    return Ok(true);
}

fn find_eye_center(img: &Mat) -> opencv::Result<(i32, i32)> {
    let gray_eye_f64 = convert_to_gray(img)?;

    // We calculate first the gradient's x-component
    let mut x_grad = compute_gradient(&gray_eye_f64)?;

    let mut t_gray_img_f64 = Mat::default();
    core::transpose(&gray_eye_f64, &mut t_gray_img_f64)?;
    let mut y_grad = Mat::default();
    core::transpose(&compute_gradient(&t_gray_img_f64)?, &mut y_grad)?;

    // Compute gradient's magnitude
    let gradient_mat = compute_magnitude(&x_grad, &y_grad)?;
    /*
        let mut display_grad = Mat::default();
        core::normalize(
            &x_grad,
            &mut display_grad,
            0.,
            255.,
            core::NORM_MINMAX,
            core::CV_8U,
            &core::no_array(),
        )?;
        highgui::imshow("gradiente", &display_grad)?;
        highgui::wait_key(0)?;
    */

    // SHOULD I REALLY NEED THIS?
    // Gradient treshold

    let treshold = compute_dynamic_treshold(&gradient_mat, GRADIENT_TRESHOLD)?;

    for row in 0..img.rows() {
        for col in 0..img.cols() {
            let magnitude = *gradient_mat.at_2d::<f64>(row, col)?;
            if magnitude > treshold {
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
    // SHOULD I REALLY NEED THIS? ^
    /*
        let mut display_x_grad = Mat::default();
        core::normalize(
            &x_grad,
            &mut display_x_grad,
            0.,
            255.,
            core::NORM_MINMAX,
            core::CV_8U,
            &core::no_array(),
        )?;
        highgui::imshow("x_grad", &display_x_grad)?;
        highgui::wait_key(0)?;

        let mut display_y_grad = Mat::default();
        core::normalize(
            &y_grad,
            &mut display_y_grad,
            0.,
            255.,
            core::NORM_MINMAX,
            core::CV_8U,
            &core::no_array(),
        )?;
        highgui::imshow("y_grad", &display_y_grad)?;
        highgui::wait_key(0)?;
    */
    let mut weight = Mat::default();
    imgproc::gaussian_blur(
        &gray_eye_f64,
        &mut weight,
        core::Size::new(7, 7), // or (3, 3)
        0.,
        0.,
        core::BORDER_DEFAULT,
    )?;

    let mut display_img = Mat::default();
    core::normalize(
        &gray_eye_f64,
        &mut display_img,
        0.,
        255.,
        core::NORM_MINMAX,
        core::CV_8U,
        &core::no_array(),
    )?;
    highgui::imshow("weight", &display_img)?;
    highgui::wait_key(0)?;

    let mut display_weight = Mat::default();
    core::normalize(
        &weight,
        &mut display_weight,
        0.,
        255.,
        core::NORM_MINMAX,
        core::CV_8U,
        &core::no_array(),
    )?;
    highgui::imshow("weight", &display_weight)?;
    highgui::wait_key(0)?;

    for row in 0..weight.rows() {
        for col in 0..weight.cols() {
            let value = *weight.at_2d::<f64>(row, col)?;
            *weight.at_2d_mut::<f64>(row, col)? = 255. - value;
        }
    }

    let mut display_weight = Mat::default();
    core::normalize(
        &weight,
        &mut display_weight,
        0.,
        255.,
        core::NORM_MINMAX,
        core::CV_8U,
        &core::no_array(),
    )?;
    highgui::imshow("weight", &display_weight)?;
    highgui::wait_key(0)?;

    let mut result = Mat::new_rows_cols_with_default(
        weight.rows(),
        weight.cols(),
        core::CV_64F,
        core::Scalar::all(0.),
    )?;

    for row in 0..weight.rows() {
        for col in 0..weight.cols() {
            test_possibile_centers_formula(row, col, &weight, &mut result, &x_grad, &y_grad)?;
        }
    }

    let num_gradients = weight.rows() * weight.cols();
    for row in 0..result.rows() {
        for col in 0..result.cols() {
            let value = *result.at_2d::<f64>(row, col)?;
            *result.at_2d_mut::<f64>(row, col)? = value * (1. / num_gradients as f64);
        }
    }

    let mut display_result = Mat::default();
    core::normalize(
        &result,
        &mut display_result,
        0.,
        255.,
        core::NORM_MINMAX,
        core::CV_8U,
        &core::no_array(),
    )?;
    highgui::imshow("result", &display_result)?;
    highgui::wait_key(0)?;

    let mut min_val = 0f64;
    let mut max_val = 0f64;
    let mut min_loc = core::Point::new(0, 0);
    let mut max_loc = core::Point::new(0, 0);
    core::min_max_loc(
        &result,
        Some(&mut min_val),
        Some(&mut max_val),
        Some(&mut min_loc),
        Some(&mut max_loc),
        &core::no_array(),
    );

    let mut mask = Mat::default();
    imgproc::threshold(
        &result,
        &mut mask,
        max_val * 0.9,
        1.,
        imgproc::THRESH_BINARY,
    );

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
    highgui::imshow("mask", &display_mask)?;
    highgui::wait_key(0)?;

    return Ok((1, 1));
}