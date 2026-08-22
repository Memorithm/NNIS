fn main() {
    println!("cargo:rustc-check-cfg=cfg(nnis_docs_rs)");
    if std::env::var("CARGO_CFG_DOC").is_ok() && std::env::var("DOCS_RS").is_ok() {
        println!("cargo:rustc-cfg=nnis_docs_rs");
    }
}
