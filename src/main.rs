use std::io::{self, Write};
use std::path::Path;
use std::fs;
use image::{RgbImage, Rgb, ImageBuffer};
use rayon::prelude::*;

fn generate_box_blur_kernel(size: usize) -> Vec<Vec<f32>> {
    // Filled with 1/(n*n) to average (blur) neighboring pixels
    // Kernel size must be odd to ensure the center pixel is included.
    // Kernel is normalized to ensure output has the same brightness as the input.
    let value = 1.0 / (size * size) as f32;
    vec![vec![value; size]; size]
}

/// Applies an (n x n) convolution kernel to an RGB image using multi-threading.
/// Each color channel (R, G, B) is processed independently.
fn apply_convolution(image: &RgbImage, kernel: &Vec<Vec<f32>>) -> RgbImage {
    let (width, height) = image.dimensions();
    let kernel_size = kernel.len();
    let half_k = kernel_size as i32 / 2;

    // Create an empty output image with the same dimensions
    let mut output: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    // Uses Rayon to parallelize row processing
    output
        .enumerate_rows_mut()
        .par_bridge() // Convert to parallel iterator
        .for_each(|(_y, row)| {
            for (x, _y, pixel) in row {
                let mut sum_r = 0.0;
                let mut sum_g = 0.0;
                let mut sum_b = 0.0;

                // Applies the kernel over the pixel neighborhood
                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let nx = (x as i32 + kx as i32 - half_k).clamp(0, (width - 1) as i32) as u32;
                        let ny = (_y as i32 + ky as i32 - half_k).clamp(0, (height - 1) as i32) as u32;

                        let neighbor_pixel = image.get_pixel(nx, ny);
                        sum_r += neighbor_pixel[0] as f32 * kernel[ky][kx];
                        sum_g += neighbor_pixel[1] as f32 * kernel[ky][kx];
                        sum_b += neighbor_pixel[2] as f32 * kernel[ky][kx];
                    }
                }

                // Clamp values and assign them to the output pixel
                pixel.0[0] = sum_r.round().clamp(0.0, 255.0) as u8;
                pixel.0[1] = sum_g.round().clamp(0.0, 255.0) as u8;
                pixel.0[2] = sum_b.round().clamp(0.0, 255.0) as u8;
            }
        });

    output
}

/// Blurs an image using a dynamically generated box blur kernel with multi-threading.
fn blur_image(input_path: &str, output_path: &str, blur_size: usize) {
    // Ensure the kernel size is odd (required for centering)
    let blur_size = if blur_size % 2 == 0 { blur_size + 1 } else { blur_size };

    // Load the image and convert it to RGB format
    let image = image::open(input_path)
        .expect("Failed to open image")
        .into_rgb8(); // Convert to RGB format

    // Generate the blur kernel dynamically
    let kernel = generate_box_blur_kernel(blur_size);

    // Apply the blur using convolution (multi-threaded)
    let blurred_image = apply_convolution(&image, &kernel);

    // Save the blurred image
    blurred_image.save(output_path)
        .expect("Failed to save blurred image");

    println!("Blurred image saved to '{}'", output_path);
}

fn generate_sharpen_kernel() -> Vec<Vec<f32>> {
    // This kernel is used to enhance edges in an image.
    // The center pixel is given a higher weight (5.0) to make it stand out more, while the neighboring pixels
    // are given a negative weight (-1.0) to reduce their influence, effectively highlighting edges.
    vec![
        vec![0.0, -1.0,  0.0],
        vec![-1.0, 5.0, -1.0],
        vec![0.0, -1.0,  0.0],
    ]
}

/// Sharpens an image using a convolutional sharpening filter.
fn sharpen_image(input_path: &str, output_path: &str) {
    // Load the image and convert it to RGB format
    let image = image::open(input_path)
        .expect("Failed to open image")
        .into_rgb8();

    // Generate the sharpening kernel
    let kernel = generate_sharpen_kernel();

    // Apply the sharpening filter using convolution
    let sharpened_image = apply_convolution(&image, &kernel);

    // Save the sharpened image
    sharpened_image.save(output_path)
        .expect("Failed to save sharpened image");

    println!("Sharpened image saved to '{}'", output_path);
}


fn find_image() -> Option<String> {
    let input_dir = "images/";
    if !Path::new(input_dir).exists() {
        println!("Error: '{}' folder does not exist. Please create it and add a .jpg file.", input_dir);
        return None;
    }

    fs::read_dir(input_dir).ok()?.find_map(|entry| {
        let path = entry.ok()?.path();
        if path.is_file() && matches!(path.extension()?.to_str()?, "jpg" | "jpeg") {
            Some(path.to_str()?.to_string())
        } else {
            None
        }
    }).or_else(|| {
        println!("Error: No .jpg files found in '{}'. Please add an image and try again.", input_dir);
        None
    })
}

fn main() {
    
    let image_path = match find_image() {
        Some(file) => file,
        None => {
            eprintln!("Error: No '.jpg' found in 'images/'. Exiting.");
            return;
        },
    };

    println!("Please enter 1 for Blur or 2 for Sharpen.");
    io::stdout().flush().unwrap(); // Ensure is displayed immediately

    let mut choice = String::new();
    io::stdin().read_line(&mut choice).expect("Failed to read input.");
    let choice = choice.trim(); // Remove newline

    let modified: String;
    
    if choice == "1" {

        print!("Enter blur strength (odd value, i.e., 3, 5, 7): ");
        io::stdout().flush().unwrap();

        let mut blur_strength = String::new();
        io::stdin().read_line(&mut blur_strength).expect("Failed to read input.");
        
        // Convert to usize (default to 5 if invalid)
        let blur_strength: usize = blur_strength.trim().parse().unwrap_or(5);

        // Ensure blur strength is odd 
        let blur_strength = if blur_strength % 2 == 0 { blur_strength + 1 } else { blur_strength };

        modified = format!("images/{}_blurred_{}.jpg", Path::new(&image_path).file_stem().expect("Failed to get file stem").to_str().expect("Failed to convert to str"), blur_strength);

        println!("Applying blur with strength {}...", blur_strength);
        blur_image(&image_path, &modified, blur_strength);

    } else if choice == "2" {
        modified = format!("images/{}_sharpened.jpg", Path::new(&image_path).file_stem().expect("Failed to get file stem").to_str().expect("Failed to convert to str"));

        println!("Sharpening the image...");
        sharpen_image(&image_path, &modified);
    
    } else {
        println!("Invalid choice! Please enter 1 for Blur or 2 for Sharpen.");
        return;
    }

    println!("Processing complete. Output saved as '{}'", modified);

}


