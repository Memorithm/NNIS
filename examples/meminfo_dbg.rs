fn main() {
    let dev = match nnis_rt::Device::first() { Ok(d) => d, Err(e) => { println!("no device: {e}"); return; } };
    let ctx = nnis_rt::Context::new(&dev).unwrap();
    println!("ctx retained, ordinal={}", ctx.device_ordinal());
    match ctx.mem_info() {
        Ok((f, t)) => println!("mem ok: free={f} total={t}"),
        Err(e) => println!("mem_info failed: {e:?}"),
    }
}
