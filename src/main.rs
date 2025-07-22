mod find_eye_center;
mod utils;

use opencv::{
    core::{self, Scalar},
    highgui,
    imgproc::{self, LINE_8},
    objdetect,
    prelude::*,
    types, videoio,
};

fn main() -> opencv::Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    // Load the pre-trained face classifier
    let mut face_detector = utils::helpers::load_detector(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    )?;

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        let mut gray = Mat::default();
        imgproc::cvt_color(
            &frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        );
        let mut faces = types::VectorOfRect::new();

        face_detector.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            2,
            objdetect::CASCADE_SCALE_IMAGE,
            core::Size::new(150, 150),
            core::Size::new(0, 0),
        )?;

        if faces.len() > 0 {
            for face in faces {
                if utils::constants::K_SMOOTH_FACE_IMAGE {
                    let sigma = utils::constants::K_SMOOTH_FACE_FACTOR * face.width as f64;
                    imgproc::gaussian_blur(
                        &gray.clone(),
                        &mut gray,
                        core::Size::new(0, 0),
                        sigma,
                        sigma,
                        core::BORDER_DEFAULT,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                }

                let eye_region_width = (face.width as f64
                    * (utils::constants::K_EYE_PERCENT_WIDTH as f64 / 100.))
                    as i32;
                let eye_region_height = (face.width as f64
                    * (utils::constants::K_EYE_PERCENT_HEIGHT as f64 / 100.))
                    as i32;
                let eye_region_top = (face.height as f64
                    * (utils::constants::K_EYE_PERCENT_TOP as f64 / 100.))
                    as i32;
                let left_eye_region = core::Rect_::new(
                    ((face.width as f64) * (utils::constants::K_EYE_PERCENT_SIDE as f64 / 100.))
                        as i32,
                    eye_region_top,
                    eye_region_width,
                    eye_region_height,
                );

                let right_eye_region = core::Rect_::new(
                    ((face.width as f64 - eye_region_width as f64)
                        - face.width as f64 * (utils::constants::K_EYE_PERCENT_SIDE as f64 / 100.))
                        as i32,
                    eye_region_top,
                    eye_region_width,
                    eye_region_height,
                );

                //-- Find Eye Centers
                let mut face_roi = utils::helpers::compute_roi(&gray, &face)?;
                let mut left_pupil = find_eye_center::find_eye_center(&face_roi, &left_eye_region)?;
                let mut right_pupil =
                    find_eye_center::find_eye_center(&face_roi, &right_eye_region)?;

                right_pupil.x += right_eye_region.x;
                right_pupil.y += right_eye_region.y;
                left_pupil.x += left_eye_region.x;
                left_pupil.y += left_eye_region.y;

                imgproc::rectangle(
                    &mut face_roi,
                    left_eye_region,
                    Scalar::new(255f64, 0f64, 0f64, 0f64),
                    1,
                    imgproc::LINE_8,
                    0,
                );

                imgproc::rectangle(
                    &mut face_roi,
                    right_eye_region,
                    Scalar::new(255f64, 0f64, 0f64, 0f64),
                    1,
                    imgproc::LINE_8,
                    0,
                );

                imgproc::circle(
                    &mut face_roi,
                    left_pupil,
                    2,
                    Scalar::new(255f64, 0f64, 0f64, 0f64),
                    1,
                    imgproc::LINE_8,
                    0,
                );

                imgproc::circle(
                    &mut face_roi,
                    right_pupil,
                    2,
                    Scalar::new(255f64, 0f64, 0f64, 0f64),
                    1,
                    imgproc::LINE_8,
                    0,
                );
                highgui::imshow("gray face", &face_roi);
            }

            let key = highgui::wait_key(10)?;
            if key > 0 && key != 255 {
                break;
            }
        }
    }
    Ok(())
}
