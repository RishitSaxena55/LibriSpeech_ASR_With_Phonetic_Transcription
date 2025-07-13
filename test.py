def test(test_decoder, test_beam_width, model, test_loader):
    results = []
    model.eval()
    print("Testing")
    for data in tqdm(test_loader):
        x, lx = data
        x = x.to(device)

        with torch.no_grad():
            h, lh = model(x, lx)

        prediction_string = decode_prediction(h, lh.to(device), test_decoder, LABELS)
        results.extend(prediction_string)

        del x, lx, h, lh
        torch.cuda.empty_cache()

