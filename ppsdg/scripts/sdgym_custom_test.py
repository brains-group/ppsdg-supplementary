import sdgym

def train_synthesizer (data, metadata):
    print(data, metadata)
    return (data, metadata)

def sample_synthesizer (state, nrows):
    print(nrows)
    return state[0][:nrows]

CustomSynthesizer = sdgym.create_single_table_synthesizer(
    get_trained_synthesizer_fn = train_synthesizer,
    sample_from_synthesizer_fn = sample_synthesizer,
    display_name="CustomSynthesizer",
)

sdgym.benchmark_single_table(
    custom_synthesizers=[CustomSynthesizer],
    sdv_datasets=["fake_hotel_guests"],
)
