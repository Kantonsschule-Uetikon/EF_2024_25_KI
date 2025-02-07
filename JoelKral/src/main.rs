extern crate core;

use std::alloc;
use std::cell::RefCell;
use glium::backend::glutin::SimpleWindowBuilder;
use glium::backend::Backend;
use glium::buffer::Content;
use glium::glutin::surface::WindowSurface;
use glium::program::ComputeShader;
use glium::texture::{MipmapsOption, RawImage2d, UncompressedFloatFormat};
use glium::uniforms::{ImageUnitAccess, ImageUnitFormat, MagnifySamplerFilter, MinifySamplerFilter, SamplerWrapFunction, UniformBuffer};
use glium::winit::application::ApplicationHandler;
use glium::winit::event::WindowEvent;
use glium::winit::event_loop::{ActiveEventLoop, EventLoop};
use glium::winit::window::{Window, WindowAttributes, WindowId};
use glium::{implement_buffer_content, implement_uniform_block, uniform, Display, Surface, Texture2d};
use itertools::Itertools;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::mem::offset_of;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use image::metadata::Orientation;

fn main() {
    let event_loop = EventLoop::builder().build().unwrap();
    let mut app = Application::default();
    let _ = event_loop.run_app(&mut app).unwrap();
}





#[derive(Default)]
struct Application {
    window: Option<Window>,
    display: Option<Display<WindowSurface>>,
    shaders: BTreeMap<String, ComputeShader>,

    result: Option<Texture2d>,
    handler: NeuralHandler,
}
#[derive(Default)]
struct NeuralHandler {
    weights: Box<Vec<UniformBuffer<Weights>>>,
    temp_weights: RefCell<Box<Vec<UniformBuffer<Weights>>>>,
    training_data: Vec<[Texture2d;2]>,
    test_data: Vec<[Texture2d;2]>,
    layers: usize,
}
#[derive(Copy)]
#[derive(Clone)]
struct Dst {
    dst: [f32;3]
}
implement_uniform_block!(Dst,dst);
#[derive(Clone, Copy)]
#[derive(Debug)]
struct Weights {
    weights: [f32; ITEMS_PER_LAYER],
}
impl<> ::glium::uniforms::UniformBlock for Weights<> {
    fn matches(layout: &::glium::program::BlockLayout, base_offset: usize)
               -> ::std::result::Result<(), ::glium::uniforms::LayoutMismatchError>
    {
        use std::mem;
        use ::glium::program::BlockLayout;
        use ::glium::uniforms::LayoutMismatchError;

        if let &BlockLayout::Struct { ref members } = layout {
            for &(ref name, _) in members {
                if name != "weights" && true {
                    return Err(LayoutMismatchError::MissingField {
                        name: name.clone(),
                    });
                }
            }

            fn matches_from_ty<T:?Sized>(_: &T,
                                         layout: &::glium::program::BlockLayout, base_offset: usize)
                                         -> ::std::result::Result<(), ::glium::uniforms::LayoutMismatchError>
            {
                <Weights>::matches(layout, base_offset)
            }

            let reflected_ty = members.iter().find(|&&(ref name, _)| {
                name == "weights"
            });
            let reflected_ty = match reflected_ty {
                Some(t) => &t.1,
                None => return Err(LayoutMismatchError::MissingField {
                    name: "weights".to_owned(),
                })
            };
            let dummy: *const Weights = unsafe { mem::zeroed() };
            let input_offset = {
                let possibly_fat_pointer_to_field = unsafe { &(*dummy).weights };
                let pointer_to_possibly_fat_pointer_to_field: &u64 = unsafe { mem::transmute(&possibly_fat_pointer_to_field) };
                let pointer_to_field = *pointer_to_possibly_fat_pointer_to_field;
                pointer_to_field as usize
            };

            match matches_from_ty(unsafe { &(*dummy).weights }, reflected_ty, input_offset) {
                Ok(_) => (),
                Err(e) => return Err(LayoutMismatchError::MemberMismatch {
                    member: "weights".to_owned(),
                    err: Box::new(e),
                })
            };

            Ok(())
        } else {
            Err(LayoutMismatchError::LayoutMismatch {
                expected: layout.clone(),
                obtained: <Self as ::glium::uniforms::UniformBlock>::build_layout(base_offset),
            })
        }
    }

    fn build_layout(base_offset: usize) -> ::glium::program::BlockLayout {
        use ::glium::program::BlockLayout;

        fn layout_from_ty<T:?Sized>(_: Option<&T>, base_offset: usize)
                                    -> BlockLayout
        {
            <Weights>::build_layout(base_offset)
        }

        BlockLayout::Struct {
            members: <[_]>::into_vec(
                Box::new([(
                    "weights".to_owned(),
                    {
                        let offset = {
                            { offset_of!(Weights, weights) }
                        };
                        let field_option = None::<&Weights>.map(|v| &v.weights);
                        layout_from_ty(field_option, offset + base_offset)
                    }
                )])
            ),
        }
    }
}

const IMAGE_COUNT: usize = 1;
const HELP: f64 = 1000000f64;
const MARGIN: f64 = 0.1;
const RATE: f32 = 1.0;
const LAYERS: usize = 1;

const IMAGE_SIZE: usize = 400;
const UNIT_SIZE: usize = 10;
const ITEMS_PER_LAYER: usize = 3 * IMAGE_SIZE.pow(2) / UNIT_SIZE.pow(2);

impl Application {

    fn train(&mut self, margin: f64, rate: f32) {
        let mut rng = rand::rng();
        let mut highest_precision = 0f64;
        let mut iterations = 0;
        'outer: loop {
            iterations += 1;
            //create a throwaway list containing references(!) to the training data
            //then, shuffle that thang - O(n) operation; manageable I'd guess
            let mut throwaway = self.handler.training_data.iter().collect_vec();
            throwaway.shuffle(&mut rng);
            while highest_precision < (1f64 - margin) && !throwaway.is_empty() {
                //pop last image and corresponding test from the throwaway list - pop is O(1); nice
                let [image, test] = throwaway.pop().unwrap();
                //adjust the weights stored in handler.temp_weights
                self.adjust_weights(rate, &mut rng);
                //run one image through the network
                let result = self.process_image(&image, &self.handler.temp_weights.borrow());
                //compare the result against the test
                let mut precision = self.test_image(&result, &test);
                println!("precision: {}",precision);
                if iterations > 1000 {
                    precision = 1f64;
                }
                if precision > highest_precision {
                    highest_precision = precision;
                    std::mem::swap(&mut self.handler.weights, &mut self.handler.temp_weights.get_mut());
                    if highest_precision >= (1f64 - margin) { println!("\n\n\n\n\n WE DONESO \n\n\n\n"); break 'outer }
                }
            }
        }

    }



    fn process_image(&self, image: &Texture2d, weights: &Vec<UniformBuffer<Weights>>) -> Texture2d {
        //CREATE EMPTY SOURCE AND DESTINATION TEXTURES
        let src = Texture2d::empty_with_format(
            self.display.as_ref().unwrap(),
            UncompressedFloatFormat::U8U8U8U8,
            MipmapsOption::NoMipmap,
            IMAGE_SIZE as u32,
            IMAGE_SIZE as u32).unwrap();
        let dst = Texture2d::empty_with_format(
            self.display.as_ref().unwrap(),
            UncompressedFloatFormat::U8U8U8U8,
            MipmapsOption::NoMipmap,
            IMAGE_SIZE as u32,
            IMAGE_SIZE as u32).unwrap();
        //FILL STARTING IMAGE INTO SOURCE AND CLEAR DESTINATION
        image.as_surface().fill(&src.as_surface(), MagnifySamplerFilter::Nearest);
        dst.as_surface().clear_color(0.0,0.0,0.0,1.0);
        //ITERATE OVER ALL WEIGHTS
        weights.iter().for_each(|weight| {
            //RUN COMPUTE SHADER FOR EVERY PIXEL IN SOURCE
            self.shaders["aleph"].execute(
                uniform!{
                    src: src.sampled(),
                    dst: dst.image_unit(ImageUnitFormat::RGBA8).unwrap().set_access(ImageUnitAccess::Write),
                    buf: weight,
                },
                IMAGE_SIZE as u32,
                IMAGE_SIZE as u32,
                1);
            //FILL DESTINATION INTO SOURCE, THEN CLEAR DESTINATION
            dst.as_surface().fill(&src.as_surface(), MagnifySamplerFilter::Nearest);
            dst.as_surface().clear_color(0.0,0.0,0.0,1.0);
        });
        //RETURN SOURCE TEXTURE, CONTAINING THE LAST SHADER OUTPUT
        src
    }

    fn test_image(&self, image: &Texture2d, test: &Texture2d) -> f64 {
        /*let mut dst = UniformBuffer::new(self.display.as_ref().unwrap(), Dst { dst: [0f32;3] }).unwrap();
        self.shaders["beth"].execute(
            uniform!{
                src1: image.sampled(),
                src2: test.sampled(),
                dst: &*dst,
            },
            image.width(),image.height(),1);
        let result: [f32;3] = dst.read().unwrap().dst;
        (3f64+HELP)/(result[0]+result[1]+result[2]+1f32) as f64*/
        0f64
    }


    fn adjust_weights<R: Rng + ?Sized>(&self, rate: f32, rng: &mut R) {
        self.handler.weights.iter()
            .zip(self.handler.temp_weights.borrow_mut().iter_mut())
            .for_each(|(buf,tmp)| {
                let mut values = buf.read().unwrap().weights;
                values.iter_mut().for_each(|val| {
                    //literally just   +-0.5 * learning_rate
                    *val += rate * ( rng.random::<f32>() / 2f32 - 1f32 )
                });
                tmp.map_write().write(Weights { weights: values });
            });
    }

    fn load_images(&self, display: &Display<WindowSurface>, dir_path: &PathBuf, cat: &str, dimensions: (u32,u32)) -> Vec<Texture2d> {
        let mut dir = std::fs::read_dir(&dir_path.join(cat))
            .expect(format!("Couldn't find {} data", cat).as_str());
        let mut data: Vec<Texture2d> = Vec::with_capacity(dir.by_ref().count());
        let mut dir = std::fs::read_dir(&dir_path.join(cat))
            .expect(format!("Couldn't find {} data", cat).as_str());
        let dir = dir.take(IMAGE_COUNT);
        dir.for_each(|entry| {
            let entry = entry.unwrap().path();
            let mut image = image::open(&entry).unwrap();
            image.apply_orientation(Orientation::FlipVertical);
            let image = image.into_rgb8();
            let image_dimensions = image.dimensions();
            if image_dimensions != dimensions { println!("{:?},{:?},{:?}",image_dimensions, dimensions, entry);  panic!("input data dimension mismatch!") }
            let texture = Texture2d::with_format(
                display,
                RawImage2d::from_raw_rgb(image.into_raw(), image_dimensions),
                UncompressedFloatFormat::F32F32F32,
                MipmapsOption::NoMipmap).expect(format!("Failed to load {} image", cat).as_str());
            data.push(texture);
        });
        data
    }

    fn init_handler(&mut self, data: PathBuf, layers: usize, dimensions: (u32,u32)) {
        if let Some(display) = self.display.as_ref() {
            println!("loading training data");
            let training_data = self.load_images(display, &data, "data", dimensions);
            println!("loading control data");
            let control_data = self.load_images(display, &data, "control", dimensions);
            println!("loading test training data");
            let test_data = self.load_images(display, &data, "test/data", dimensions);
            println!("loading test control data");
            let test_control_data = self.load_images(display, &data, "test/control", dimensions);
            assert_eq!(training_data.is_empty(),false);
            println!("zipping training data");
            let training_data = training_data.into_iter().zip(control_data).map(|tuple|{ [tuple.0,tuple.1] }).collect_vec();
            println!("zipping test data");
            let test_data = test_data.into_iter().zip(test_control_data).map(|tuple|{ [tuple.0,tuple.1] }).collect_vec();
            assert_eq!(training_data.is_empty(),false);
            let mut weights: Vec<UniformBuffer<Weights>> = Vec::with_capacity(layers);
            let mut temp_weights: Vec<UniformBuffer<Weights>> = Vec::with_capacity(layers);
            while weights.len() < weights.capacity() {
                weights.push(UniformBuffer::new(self.display.as_ref().unwrap(), Weights { weights: [0.5f32;ITEMS_PER_LAYER] }).unwrap());
                temp_weights.push(UniformBuffer::new(self.display.as_ref().unwrap(), Weights { weights: [0.5f32;ITEMS_PER_LAYER] }).unwrap());
            }
            self.handler = NeuralHandler {
                weights: Box::new(weights),
                temp_weights: RefCell::new(Box::new(temp_weights)),
                training_data,
                test_data,
                layers,
            }
        }}
}


impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attributes = WindowAttributes::default()
            .with_title("yuh")
            .with_transparent(false);
        let (window, display) = SimpleWindowBuilder::new()
            .set_window_builder(attributes).build(event_loop);
        (self.window, self.display) = (Some(window), Some(display));
        if let Some(display) = self.display.as_ref() {
            std::fs::read_dir(r"./src/shader").unwrap().for_each(|entry| {
                let mut source: String = String::new();
                let entry = entry.as_ref().unwrap();
                File::open(entry.path()).unwrap()
                    .read_to_string(&mut source).unwrap();
                self.shaders.insert(entry.file_name().to_str().unwrap().strip_suffix(".comp").unwrap().to_owned(),ComputeShader::from_source(display, &*source).unwrap());
            })}
        let (width, height) = self.display.as_ref().unwrap().get_framebuffer_dimensions();
        let result_texture = Texture2d::empty_with_format(
            self.display.as_ref().unwrap(),
            UncompressedFloatFormat::U8U8U8U8,
            MipmapsOption::NoMipmap,
            width, height).unwrap();
        result_texture.as_surface().clear_color(0.0,0.0,0.0,1.0);
        self.result = Some(result_texture);




        println!("\n\n\n\n\n\n INITIATION \n\n\n\n\n");
        self.init_handler(PathBuf::from(r"./src/colorization"), LAYERS, (IMAGE_SIZE as u32, IMAGE_SIZE as u32));
        println!("\n\n\n\n\n\n TRAINING \n\n\n\n\n");
        self.train(MARGIN, RATE);
        println!("\n\n\n\n\n\n UTILIZATION \n\n\n\n\n");
        let random_test_data = self.handler.test_data.choose(&mut rand::rng()).unwrap();
        let test_result = self.process_image(&random_test_data[0], &self.handler.weights);
        test_result.as_surface().fill(&self.result.as_ref().unwrap().as_surface(), MagnifySamplerFilter::Linear);
        println!("\n\n\n\n\n\n EVENT LOOP REACHED \n\n\n\n\n");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.display.as_ref().unwrap().resize(size.into());
                self.window.as_ref().unwrap().request_redraw();
            },
            WindowEvent::RedrawRequested => {
                if let (Some(window), Some(display)) = (self.window.as_ref(),self.display.as_ref()) {

                    let frame = display.draw();

                    self.result.as_ref().unwrap().as_surface().fill(&frame, MagnifySamplerFilter::Nearest);

                    frame.finish().unwrap();
                    window.request_redraw();
                }}
            _ => ()
        }
    }
}



