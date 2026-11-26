from anchors import AnchorGenerator

anchor_gen = AnchorGenerator(input_size=640)
anchors = anchor_gen.to_tensor('cpu')
print(f"Anchor 数量: {anchors.shape[0]}")
print(f"Anchor X 范围: [{anchors[:,0].min():.0f}, {anchors[:,2].max():.0f}]")
print(f"Anchor Y 范围: [{anchors[:,1].min():.0f}, {anchors[:,3].max():.0f}]")