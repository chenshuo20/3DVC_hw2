import open3d as o3d

pcd = o3d.io.read_point_cloud("bunny.ply")

vis = o3d.visualization.Visualizer()

vis.create_window(visible=False)
vis.add_geometry(pcd)

vis.poll_events()
vis.update_renderer()
image = vis.capture_screen_float_buffer()

o3d.io.write_image("output.png", image)

vis.destroy_window()
