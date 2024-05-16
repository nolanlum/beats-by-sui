extern crate byteorder;
extern crate symphonia;

use std::{
    io::{self, Write},
    time::Instant,
};

use beats_by_sui::{Beatmania, CoarseBeatDetector};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{self, DecoderOptions},
    errors::Error,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

fn main() {
    // Get the first command line argument.
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("file path not provided");

    let start = Instant::now();

    // Open the media source.
    let src = std::fs::File::open(path).expect("failed to open media");

    // Create the media source stream.
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    // Use the default options for metadata and format readers.
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    // Probe the media source.
    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), mss, &fmt_opts, &meta_opts)
        .expect("unsupported format");

    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != codecs::CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    // Use the default options for the decoder.
    let dec_opts: DecoderOptions = Default::default();

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("unsupported codec");

    // Store the track identifier, it will be used to filter packets.
    let track_id = track.id;

    // Do the thing.
    let total_frame_count = track.codec_params.n_frames.unwrap();
    let sample_rate = track.codec_params.sample_rate.unwrap();
    let mut processed_frame_count = 0u64;
    let mut last_update = Instant::now();
    let mut sample_buf = None;
    let mut bpm_machine = CoarseBeatDetector::new(sample_rate);

    print!("Processed 0/{} frames", total_frame_count);
    let _ = io::stdout().flush();

    // The decode loop.
    loop {
        // Get the next packet from the media format.
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::ResetRequired) => {
                // The track list has been changed. Re-examine it and create a new set of decoders,
                // then restart the decode loop. This is an advanced feature and it is not
                // unreasonable to consider this "the end." As of v0.5.0, the only usage of this is
                // for chained OGG physical streams.
                unimplemented!();
            }
            Err(Error::IoError(_)) => {
                // Assume all IO errors are EOF.
                // Not sure how to or if we want to introspect the error...
                break;
            }
            Err(err) => {
                // A unrecoverable error occured, halt decoding.
                panic!("{}", err);
            }
        };

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples.
        match decoder.decode(&packet) {
            Ok(decoded) => {
                // If this is the *first* decoded packet, create a sample buffer matching the
                // decoded audio buffer format.
                if sample_buf.is_none() {
                    // Get the audio buffer specification.
                    let spec = *decoded.spec();

                    // Get the capacity of the decoded buffer. Note: This is capacity, not length!
                    let duration = decoded.capacity() as u64;

                    // Create the f32 sample buffer.
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                // Copy the decoded audio buffer into the sample buffer in an interleaved format.
                if let Some(buf) = &mut sample_buf {
                    processed_frame_count += decoded.frames() as u64;

                    if (Instant::now() - last_update).as_millis() > 500 {
                        print!(
                            "\rProcessed {}/{} frames",
                            processed_frame_count, total_frame_count
                        );
                        let _ = io::stdout().flush();
                        last_update = Instant::now();
                    }

                    // Sum all channels into one mono channel and pass to processor.
                    let channel_count = decoded.spec().channels.count();
                    buf.copy_interleaved_ref(decoded);
                    bpm_machine.process_samples(
                        &buf.samples()
                            .chunks_exact(channel_count)
                            .map(|x| (x.iter().sum::<f32>() / x.len() as f32))
                            .collect::<Vec<f32>>(),
                    );
                }
            }
            Err(Error::IoError(_)) => {
                // The packet failed to decode due to an IO error, skip the packet.
                continue;
            }
            Err(Error::DecodeError(_)) => {
                // The packet failed to decode due to invalid data, skip the packet.
                continue;
            }
            Err(err) => {
                // An unrecoverable error occured, halt decoding.
                panic!("{}", err);
            }
        }
    }

    let beats = bpm_machine.finalize();

    let iidx = Beatmania::new(sample_rate);
    let bpm = iidx.calculate_bpm(&beats);

    println!();
    println!("Detected tempo: {:.2} bpm", bpm);
    println!(
        "Processed {} frames ({:.2} seconds)",
        bpm_machine.processed_frames(),
        bpm_machine.processed_frames_duration().as_secs_f64()
    );
    println!(
        "Finished in {} seconds!",
        (Instant::now() - start).as_secs_f64(),
    );
    // println!("{:?}", beats);
}
