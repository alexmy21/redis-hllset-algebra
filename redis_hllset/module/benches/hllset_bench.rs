use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

// Import the HLLSet implementation
// Note: This requires the crate to expose HLLSet publicly
use redis_hllset::HLLSet;

fn bench_create_small(c: &mut Criterion) {
    let tokens: Vec<String> = (0..10).map(|i| format!("token_{}", i)).collect();
    
    c.bench_function("create_10_tokens", |b| {
        b.iter(|| {
            HLLSet::from_tokens(black_box(&tokens))
        })
    });
}

fn bench_create_large(c: &mut Criterion) {
    let tokens: Vec<String> = (0..10000).map(|i| format!("token_{}", i)).collect();
    
    c.bench_function("create_10000_tokens", |b| {
        b.iter(|| {
            HLLSet::from_tokens(black_box(&tokens))
        })
    });
}

fn bench_cardinality(c: &mut Criterion) {
    let tokens: Vec<String> = (0..1000).map(|i| format!("token_{}", i)).collect();
    let mut hll = HLLSet::from_tokens(&tokens);
    
    c.bench_function("cardinality_1000", |b| {
        b.iter(|| {
            // Clear cache to benchmark actual computation
            hll.cardinality()
        })
    });
}

fn bench_union(c: &mut Criterion) {
    let tokens_a: Vec<String> = (0..1000).map(|i| format!("a_{}", i)).collect();
    let tokens_b: Vec<String> = (500..1500).map(|i| format!("b_{}", i)).collect();
    
    let hll_a = HLLSet::from_tokens(&tokens_a);
    let hll_b = HLLSet::from_tokens(&tokens_b);
    
    c.bench_function("union_1000", |b| {
        b.iter(|| {
            hll_a.union(black_box(&hll_b))
        })
    });
}

fn bench_intersection(c: &mut Criterion) {
    let tokens_a: Vec<String> = (0..1000).map(|i| format!("a_{}", i)).collect();
    let tokens_b: Vec<String> = (500..1500).map(|i| format!("b_{}", i)).collect();
    
    let hll_a = HLLSet::from_tokens(&tokens_a);
    let hll_b = HLLSet::from_tokens(&tokens_b);
    
    c.bench_function("intersection_1000", |b| {
        b.iter(|| {
            hll_a.intersection(black_box(&hll_b))
        })
    });
}

fn bench_similarity(c: &mut Criterion) {
    let tokens_a: Vec<String> = (0..1000).map(|i| format!("a_{}", i)).collect();
    let tokens_b: Vec<String> = (500..1500).map(|i| format!("b_{}", i)).collect();
    
    let mut hll_a = HLLSet::from_tokens(&tokens_a);
    let mut hll_b = HLLSet::from_tokens(&tokens_b);
    
    c.bench_function("jaccard_similarity_1000", |b| {
        b.iter(|| {
            hll_a.jaccard_similarity(black_box(&mut hll_b))
        })
    });
}

fn bench_content_key(c: &mut Criterion) {
    let tokens: Vec<String> = (0..100).map(|i| format!("token_{}", i)).collect();
    
    c.bench_function("content_key_100_tokens", |b| {
        b.iter(|| {
            HLLSet::content_key(black_box(&tokens))
        })
    });
}

fn bench_serialization(c: &mut Criterion) {
    let tokens: Vec<String> = (0..1000).map(|i| format!("token_{}", i)).collect();
    let hll = HLLSet::from_tokens(&tokens);
    
    c.bench_function("serialize_1000", |b| {
        b.iter(|| {
            hll.to_bytes()
        })
    });
    
    let bytes = hll.to_bytes();
    c.bench_function("deserialize_1000", |b| {
        b.iter(|| {
            HLLSet::from_bytes(black_box(&bytes))
        })
    });
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");
    
    for size in [100, 1000, 10000, 100000].iter() {
        let tokens: Vec<String> = (0..*size).map(|i| format!("token_{}", i)).collect();
        
        group.bench_with_input(
            BenchmarkId::new("create", size),
            size,
            |b, _| {
                b.iter(|| HLLSet::from_tokens(black_box(&tokens)))
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_create_small,
    bench_create_large,
    bench_cardinality,
    bench_union,
    bench_intersection,
    bench_similarity,
    bench_content_key,
    bench_serialization,
    bench_scaling,
);

criterion_main!(benches);
