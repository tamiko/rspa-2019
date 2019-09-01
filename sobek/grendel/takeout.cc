    if (geometry_ == "circle") {

      const auto center = Point<dim>(0.5, 0.5);

      manifold_.reset(new SphericalManifold<dim>(center));
      triangulation.set_manifold(1, *manifold_);

      const double a = 1. + std::sqrt(2.0);
      const unsigned int n = std::max(
          2.0, 1 + std::floor(std::log(std::sqrt(2.0) / radius_) / std::log(a)));

      const auto p_00 = Point<dim>(-1, -1) / std::sqrt(2.0) / 2.;
      const auto p_10 = Point<dim>(+1, -1) / std::sqrt(2.0) / 2.;
      const auto p_01 = Point<dim>(-1, +1) / std::sqrt(2.0) / 2.;
      const auto p_11 = Point<dim>(+1, +1) / std::sqrt(2.0) / 2.;

      std::vector<Point<dim>> vertices(4 * (n + 1));

      double factor = radius_ / a;
      for (unsigned int i = 0; i < n + 1; ++i) {
        if (i == n)
          factor = std::sqrt(2.0);
        vertices[4. * i + 0] = center + p_00 * factor;
        vertices[4. * i + 1] = center + p_10 * factor;
        vertices[4. * i + 2] = center + p_01 * factor;
        vertices[4. * i + 3] = center + p_11 * factor;
        factor *= a;
      }

      std::vector<CellData<dim>> cells(1 + 4 * n, CellData<dim>());

      // center:
      {
        const auto list = std::initializer_list<unsigned int>({0, 1, 2, 3});
        std::copy(list.begin(), list.end(), cells[2 * n].vertices);
        cells[2 * n].material_id = 1;
      }

      for (unsigned int i = 0; i < n; ++i) {
        char material_id = (i == 0) ? 1 : 0;

        // bottom:
        std::vector<unsigned int> bottom = {
            4 * i + 4, 4 * i + 5, 4 * i + 0, 4 * i + 1};
        std::copy(bottom.begin(), bottom.end(), cells[i].vertices);
        cells[i].material_id = material_id;

        // left:
        std::vector<unsigned int> left = {
            4 * i + 4, 4 * i + 0, 4 * i + 6, 4 * i + 2};
        std::copy(left.begin(), left.end(), cells[n + i].vertices);
        cells[n + i].material_id = material_id;

        // right:
        std::vector<unsigned int> right = {
            4 * i + 5, 4 * i + 7, 4 * i + 1, 4 * i + 3};
        std::copy(right.begin(), right.end(), cells[2 * n + 1 + i].vertices);
        cells[2 * n + 1 + i].material_id = material_id;

        // top:
        std::vector<unsigned int> top = {
            4 * i + 6, 4 * i + 2, 4 * i + 7, 4 * i + 3};
        std::copy(top.begin(), top.end(), cells[3 * n + 1 + i].vertices);
        cells[3 * n + 1 + i].material_id = material_id;
      }

      triangulation.create_triangulation(vertices, cells, SubCellData());

      /* This is a mess: */

      triangulation.set_all_manifold_ids(0);

      auto get_it = [&triangulation](unsigned int n) {
        auto it = triangulation.begin_active();
        std::advance(it, n);
        return it;
      };

      get_it(0)->set_all_manifold_ids(1);
      get_it(n)->set_all_manifold_ids(1);
      get_it(2 * n + 1)->set_all_manifold_ids(1);
      get_it(3 * n + 1)->set_all_manifold_ids(1);
      get_it(2 * n)->set_all_manifold_ids(0);

      get_it(n - 1)->face(2)->set_boundary_id(1);
      get_it(2 * n - 1)->face(0)->set_boundary_id(0);
      get_it(3 * n)->face(2)->set_boundary_id(2);
      get_it(4 * n)->face(0)->set_boundary_id(3);

      triangulation.refine_global(refinement_);

