# df.py - implements phase two of the pipeline as a differentiable renderer
# Usage:
#   provide the target image "target.exr" in a subdirectory called "results" and run the file
import pyredner
import torch

pyredner.set_use_gpu(torch.cuda.is_available())

GRID_X = 99
GRID_Y = 270


# Materials List (index = matID)
mat_grey = pyredner.Material(
    diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5], device=pyredner.get_device()))
# The material list of the scene
materials = [mat_grey]
material_map, mesh_list, light_map = pyredner.load_obj(
    'shoe_tread.obj', return_objects=False, device=pyredner.get_device())

# Interpret mesh as grid and clone z-coordinates for optimization input, storing
#   sorting order to re-interpret later
verts = mesh_list[0][1].vertices
z_verts = verts[:, 1].clone().detach().requires_grad_(True)

order = torch.tensor([0, 2, 1], device=pyredner.get_device())
xy_coords = verts[:, [0, 2]]

grid_sorted = torch.cat((xy_coords, torch.arange(
    xy_coords.shape[0], device=pyredner.get_device()).reshape((-1, 1))), 1)
grid_sorted = grid_sorted[grid_sorted.argsort(dim=0)[:, 1]]
grid_sorted = grid_sorted[grid_sorted.sort(stable=True, dim=0)[1][:, 0]]
grid_z_ind = grid_sorted[:, 2].long()

# Construct Redner objects for scene
shape_grid = pyredner.Shape(
    vertices=torch.index_select(
        torch.cat((xy_coords, z_verts.reshape((-1, 1))), 1), 1, order),
    indices=mesh_list[0][1].indices,
    uvs=None,
    normals=None,
    material_id=0)

shapes = [shape_grid]

camera = pyredner.Camera(position=torch.tensor([0.0, 565, 0.0]),
                         look_at=torch.tensor([0.0, 0.0, 0.0]),
                         up=torch.tensor([0.0, 0.0, -1.0]),
                         fov=torch.tensor([0.19]),  # in degree
                         clip_near=1e-2,  # needs to > 0
                         resolution=(GRID_Y, GRID_X),
                         fisheye=False)

# Load lighting as uniformly white env map.
if pyredner.get_use_gpu():
    envmap = pyredner.EnvironmentMap(pyredner.imread('white.exr').cuda())
else:
    envmap = pyredner.EnvironmentMap(pyredner.imread('white.exr'))

scene = pyredner.Scene(camera, shapes, materials, envmap=envmap)

scene_args = pyredner.RenderFunction.serialize_scene(
    scene=scene,
    num_samples=1024,
    max_bounces=1)

render = pyredner.RenderFunction.apply

# Load target
target = pyredner.imread('results/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda()

# Render the initial guess
img = render(1, *scene_args)

# Save the image
pyredner.imwrite(img.cpu(), 'results/init.png')

# Compute the difference and save the images.
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/init_diff.png')

# Refine over 200 iterations
optimizer = torch.optim.Adam([z_verts], lr=1e-3)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    scene_args = pyredner.RenderFunction.serialize_scene(
        scene=scene,
        num_samples=16,  # use less samples in the Adam loop.
        max_bounces=1)
    # use a different seed every iteration to reduce bias
    img = render(t+1, *scene_args)

    # Save the intermediate render.
    pyredner.imwrite(
        img.cpu(), 'results/iter_{}.png'.format(t))
    # Pixel difference portion of loss function
    loss = (img - target).pow(2).sum()

    # Smoothing penalty:
    # |height differences|
    gridded_z = z_verts[grid_z_ind].reshape((GRID_X, GRID_Y))
    loss += (torch.abs(gridded_z[:, 0:-1] - gridded_z[:, 1:]).sum() / 2 +
             torch.abs(gridded_z[0:-1, :] - gridded_z[1:, :]).sum() / 2) * 10

    # Penalize |Normals|
    loss += (((gridded_z[:-2, :] - 2 * gridded_z[1:-1, :] + gridded_z[2:, :]).sum() / 2)
             + ((gridded_z[:-2, :] - 2 * gridded_z[1:-1, :] +
                 gridded_z[2:, :]).sum() / 2)) * 10
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()

    # Take a gradient descent step.
    optimizer.step()

    # Update z coords
    shape_grid.vertices = torch.index_select(
        torch.cat((xy_coords, z_verts.reshape((-1, 1))), 1), 1, order)

# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(
    scene=scene,
    num_samples=1024,
    max_bounces=1)
img = render(202, *scene_args)
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/final.exr')
pyredner.imwrite(img.cpu(), 'results/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(),
                 'results/final_diff.png')

# Re-interpret vector as height map and export
height_map = z_verts[grid_z_ind].reshape((GRID_X, GRID_Y)).cpu()
height_map -= height_map.min()
height_map /= height_map.max()
pyredner.imwrite(height_map, 'results/height.png')
