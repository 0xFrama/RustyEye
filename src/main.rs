use opencv::{core::{self, Rect_}, highgui, imgcodecs, imgproc, objdetect::{self, CascadeClassifier}, prelude::*, types};
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

fn load_face_detector() -> opencv::Result<CascadeClassifier> {
    let xml = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    let detector = objdetect::CascadeClassifier::new(xml)?;
    
    Ok(detector)
}

fn compute_roi(mat: &Mat, face: &Rect_<i32>) -> opencv::Result<Mat> {
    let sub_area = core::Rect::new(
                                face.x, 
                                face.y, 
                                face.width, 
                                face.height);
    let sub_mat = mat.roi(sub_area)?.clone_pointee();
    Ok(sub_mat)
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
    let mut magnitude_mat: Mat = Mat::zeros(x_grad.rows(), x_grad.cols(), core::CV_64F)?.to_mat()?; 
    core::magnitude(x_grad, y_grad, &mut magnitude_mat)?;
    Ok(magnitude_mat)
}

fn compute_dynamic_treshold(mat: &Mat, treshold: f64) -> opencv::Result<f64> {
    let mut std_magn_grad = core::Scalar::all(0.0);
    let mut mean_magn_grad = core::Scalar::all(0.0);

    core::mean_std_dev(&mat, &mut mean_magn_grad, &mut std_magn_grad, &core::no_array())?;
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

fn test_possibile_centers_formula(cx: i32, cy: i32, weight_mat: &Mat, out: &mut Mat, x_grad: &Mat, y_grad: &Mat) -> opencv::Result<()> {
    let mut dot_product_sum = 0.;
    for row in 0..out.rows() {
        for col in 0..out.cols() {
            let mut dx = (row - cx) as f64;
            let mut dy = (col - cy) as f64;

            let magnitude = ((dx * dx) + (dy * dy)).sqrt();
            dx = dx / magnitude;
            dy = dy / magnitude;

            let gX = *x_grad.at_2d::<f64>(row, col)?;
            let gY = *y_grad.at_2d::<f64>(row, col)?;

            if gX == 0. && gY == 0. {
                continue;
            }

            let mut dot_product = dx * gX + dy * gY;
            dot_product = dot_product.max(0.);
            if K_ENABLE_WEIGHT == true {
                let weight = *weight_mat.at_2d::<f64>(cx, cy)?;
                //*out.at_2d_mut::<f64>(cx, cy)? = dot_product.powi(2) * (value / K_WEIGHT_DIVISOR);
                dot_product_sum += weight * dot_product.powi(2);
            } else {
                //*out.at_2d_mut::<f64>(cx, cy)? = dot_product.powi(2);
                dot_product_sum += dot_product.powi(2);
            }
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
        0)?;

    Ok(dst)
}

fn flood_kill_edges(mat: &mut Mat) -> opencv::Result<Mat> {
    // Create mask with all 255
    let mut mask = Mat::new_size_with_default(mat.size()?, opencv::core::CV_8UC1, core::Scalar::all(255.))?;

    let mut to_do = VecDeque::new();
    to_do.push_back(core::Point::new(0, 0));

    while let Some(p) = to_do.pop_front() {
        //println!("{}", to_do.len());
        //if *mat.at_2d::<f64>(p.y, p.x)? == 0.0 {
        //    continue;
        //}
        let mut np = core::Point::new(p.x + 1, p.y);
        if flood_should_push_point(&np, mat) { to_do.push_back(np) }
        np.x = p.x - 1; np.y = p.y;
        if flood_should_push_point(&np, mat) { to_do.push_back(np) }
        np.x = p.x; np.y = p.y + 1;
        if flood_should_push_point(&np, mat) { to_do.push_back(np) }
        np.x = p.x; np.y = p.y - 1;
        if flood_should_push_point(&np, mat) { to_do.push_back(np) }
        // Kill it
        *mat.at_2d_mut::<f64>(p.y, p.x)? = 0.0;
        *mask.at_2d_mut::<u8>(p.y, p.x)? = 0;
    }

    let mut temp = Mat::default();
    mat.convert_to(&mut temp, opencv::core::CV_8UC1, 0., 0.);
    highgui::imshow("mat", &temp);
    highgui::imshow("mask", &mask);
    highgui::wait_key(0);

    Ok(mask)
}

fn flood_should_push_point(p: &core::Point, mat: &Mat) -> bool {
    p.x >= 0 && p.x < mat.cols() && p.y >= 0 && p.y < mat.rows()
}

fn find_eye_center(img: &Mat) -> opencv::Result<(i32, i32)> {
    // First, we resize the img to accelerate computation
    //let resized_img = scale_img(&img)?;

    // Then, we calculate first the gradient's x-component
    let mut x_grad = compute_gradient(&img)?;

    let mut resized_t_img = Mat::default();
    core::transpose(&img, &mut resized_t_img)?;
    let mut y_grad = Mat::default();
    core::transpose(&compute_gradient(&resized_t_img)?, &mut y_grad)?;

     // Compute gradient's magnitude
     let mut gradient_mat = compute_magnitude(&x_grad, &y_grad)?;

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

    let mut weight = Mat::default();
    imgproc::gaussian_blur(
        &img, 
        &mut weight, 
        core::Size::new(5, 5), 
        0., 
        0., 
        core::BORDER_DEFAULT)?; 

    for row in 0..weight.rows() {
        for col in 0..weight.cols() {
            let value = *weight.at_2d::<f64>(row, col)?;
            *weight.at_2d_mut::<f64>(row, col)? = 255. - value;
        }
    } 

    let mut out = Mat::new_rows_cols_with_default(weight.rows(), weight.cols(), core::CV_64F, core::Scalar::all(0.))?;
    for row in 0..weight.rows() {
        for col in 0..weight.cols() {
            
            test_possibile_centers_formula(row, col, &weight, &mut out, &x_grad, &y_grad)?;
        }
    }

    let num_gradients = weight.rows() * weight.cols();
    for row in 0..out.rows() {
        for col in 0..out.cols() {
            let value = *out.at_2d::<f64>(row, col)?;
            *out.at_2d_mut::<f64>(row, col)? = value * (1. / num_gradients as f64);
        }
    }

    let mut min_val = 0.;
    let mut max_val = 0.;
    let mut min_loc = core::Point::default();
    let mut max_loc = core::Point::default();
    core::min_max_loc(&out, Some(&mut min_val), Some(&mut max_val), Some(&mut min_loc), Some(&mut max_loc), &Mat::default())?;

    let mut floodClone = Mat::default();
    let treshold = (POST_PROCESS_THRESHOLD/100.) * max_val;
    imgproc::threshold(&out, &mut floodClone, treshold, 0., imgproc::THRESH_TOZERO);

    let mut mask = flood_kill_edges(&mut floodClone)?;

    let mut min_val = 0.;
    let mut max_val = 0.;
    let mut min_loc = core::Point::default();
    let mut max_loc = core::Point::default();
    core::min_max_loc(&out, Some(&mut min_val), Some(&mut max_val), Some(&mut min_loc), Some(&mut max_loc), &mask)?;

    Ok((max_loc.x, max_loc.y))

}


fn main() -> opencv::Result<()> {

    let mut rgb_img = imgcodecs::imread("eva.jpg", imgcodecs::IMREAD_COLOR)?;
    let gray_img_64f = convert_to_gray(&rgb_img)?;

    let mut face_detector = load_face_detector()?;
    
    loop {

        let mut faces = types::VectorOfRect::new();
        face_detector.detect_multi_scale(
            &rgb_img, 
            &mut faces, 
            1.1, 
            2, 
            objdetect::CASCADE_SCALE_IMAGE, 
            core::Size::new(30, 30), 
            core::Size::new(0,0)
        )?;

        if faces.len() > 0 {

            for face in faces.iter() {

                let gray_face = compute_roi(&gray_img_64f, &face)?;

                let eye_region_width = (gray_face.cols() as f64 * (PERC_WIDTH_EYE as f64 / 100.)) as i32; 
                let eye_region_height = (gray_face.cols() as f64 * (PERC_HEIGHT_EYE as f64 / 100.)) as i32;

                let left_eye_region = core::Rect::new(
                    face.x + (gray_face.cols() as f64 * (PERC_EYE_X as f64 / 100.)) as i32, 
                    face.y + (gray_face.rows() as f64 * (PERC_EYE_Y as f64 / 100.)) as i32, 
                    eye_region_width, 
                    eye_region_height);
      
                let right_eye_region = core::Rect::new(
                    face.x + (gray_face.cols() as f64 - eye_region_width as f64 - gray_face.cols() as f64 * (PERC_EYE_X as f64 / 100.)) as i32,
                    face.y + (gray_face.rows() as f64 * (PERC_EYE_Y as f64 / 100.)) as i32,
                    eye_region_width,
                    eye_region_height
                );           

                //let left_eye_mat_64f = compute_roi(&gray_img_64f, &left_eye_region)?;
                //let (sx_x, sx_y) = find_eye_center(&left_eye_mat_64f)?;
                
                let right_eye_mat_64f = compute_roi(&gray_img_64f, &right_eye_region)?;
                let (dx_x, dx_y) = find_eye_center(&right_eye_mat_64f)?;
            

                //let (right_x, right_y) = unscale_point(core::Point::new(dx_x, dx_y), &rgb_img)?;
                //let (left_x, left_y) = unscale_point(core::Point::new(sx_x, sx_y), &rgb_img)?;
        
                //println!("sx_x and sx_y: {} {}", right_x, right_y);
                //println!("dx_x and dx_y: {} {}", left_x, left_y);

                let mut temp = Mat::default();
                right_eye_mat_64f.convert_to(&mut temp, core::CV_8UC1, 1., 1.);

                imgproc::circle(
                    &mut temp, 
                    core::Point::new(dx_x, dx_y), 
                    1, 
                    core::Scalar::new(255f64, 0f64, 0f64, 0f64), 
                    2, 
                    0, 
                    0);

                highgui::imshow("res", &temp);
                highgui::wait_key(0);

                /*imgproc::circle(
                    &mut rgb_img, 
                    core::Point::new(left_x, left_y), 
                    1, 
                    core::Scalar::new(0f64, 255f64, 0f64, 0f64), 
                    2, 
                    0, 
                    0);*/
            }
        }
        highgui::imshow("res", &rgb_img);
        highgui::wait_key(0);
    }

    Ok(())

}
