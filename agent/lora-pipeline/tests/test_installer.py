import importlib.util
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
INSTALLER_PATH = REPO / 'extension' / 'install_extension.py'
PAYLOAD = INSTALLER_PATH.parent
CURRENT_ROUTES = REPO / 'tests/fixtures/routes-current.py'
OLD_BASE_ROUTES = REPO / 'tests/fixtures/routes-previous.py'

spec = importlib.util.spec_from_file_location('freevideo_installer', INSTALLER_PATH)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def make_source(self, route_text):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        source = Path(temp.name)
        agent = source / 'freevideo_engine' / 'agent'
        agent.mkdir(parents=True)
        (agent / 'routes.py').write_text(route_text, encoding='utf-8')
        return source, agent

    def test_check_is_read_only_and_recognizes_current_layout(self):
        source, agent = self.make_source(CURRENT_ROUTES.read_text(encoding='utf-8'))
        before = (agent / 'routes.py').read_bytes()
        result = installer.check(source, PAYLOAD)
        self.assertEqual(result['compatible'], True)
        self.assertEqual(result['installed'], False)
        self.assertEqual(len(result['source_fingerprint']), 64)
        self.assertEqual((agent / 'routes.py').read_bytes(), before)
        self.assertEqual(list(agent.iterdir()), [agent / 'routes.py'])

    def test_installs_both_supported_base_route_fixtures(self):
        for source_routes in (CURRENT_ROUTES, OLD_BASE_ROUTES):
            with self.subTest(routes=source_routes):
                source, agent = self.make_source(source_routes.read_text(encoding='utf-8'))
                original = (agent / 'routes.py').read_bytes()
                backup_root = source / 'backups'
                result = installer.install(source, PAYLOAD, backup_root)
                self.assertTrue(result['installed'])
                patched = (agent / 'routes.py').read_bytes()
                self.assertNotEqual(patched, original)
                self.assertEqual(installer.check(source, PAYLOAD)['installed'], True)
                second = installer.install(source, PAYLOAD, backup_root)
                self.assertEqual((agent / 'routes.py').read_bytes(), patched)
                self.assertEqual((agent / 'training.py').read_bytes(), (PAYLOAD / 'training.py').read_bytes())
                self.assertEqual((agent / 'training_worker.py').read_bytes(), (PAYLOAD / 'training_worker.py').read_bytes())
                self.assertNotEqual(result['backup'], second['backup'])
                self.assertEqual((Path(result['backup']) / 'routes.py').read_bytes(), original)

    def test_already_integrated_old_route_source_is_idempotent(self):
        source, agent = self.make_source(installer._prepare_routes(OLD_BASE_ROUTES.read_text(encoding='utf-8'))[0])
        before = (agent / 'routes.py').read_bytes()
        self.assertTrue(installer.check(source, PAYLOAD)['installed'] is False)
        installer.install(source, PAYLOAD, source / 'backups')
        self.assertEqual((agent / 'routes.py').read_bytes(), before)
        self.assertTrue(installer.check(source, PAYLOAD)['installed'])

    def test_partial_integration_refuses_without_writes(self):
        route_text = CURRENT_ROUTES.read_text(encoding='utf-8') + '\n# ' + installer.INTEGRATION_MARKERS[0] + '\n'
        source, agent = self.make_source(route_text)
        before = {path.name: path.read_bytes() for path in agent.iterdir()}
        result = installer.check(source, PAYLOAD)
        self.assertFalse(result['compatible'])
        self.assertIn('Partial LoRA integration', result['reason'])
        with self.assertRaises(installer.CompatibilityError):
            installer.install(source, PAYLOAD, source / 'backups')
        self.assertEqual({path.name: path.read_bytes() for path in agent.iterdir()}, before)
        self.assertFalse((source / 'backups').exists())

    def test_unknown_source_refuses_without_writing_anything(self):
        source, agent = self.make_source('from somewhere import routes\n')
        before = (agent / 'routes.py').read_bytes()
        self.assertFalse(installer.check(source, PAYLOAD)['compatible'])
        with self.assertRaises(installer.CompatibilityError):
            installer.install(source, PAYLOAD, source / 'backups')
        self.assertEqual((agent / 'routes.py').read_bytes(), before)
        self.assertEqual(list(agent.iterdir()), [agent / 'routes.py'])

    def test_backup_contains_existing_extension_modules(self):
        source, agent = self.make_source(CURRENT_ROUTES.read_text(encoding='utf-8'))
        (agent / 'training.py').write_text('old training module\n', encoding='utf-8')
        (agent / 'training_worker.py').write_text('old worker module\n', encoding='utf-8')
        old_files = {name: (agent / name).read_bytes() for name in ('routes.py', 'training.py', 'training_worker.py')}
        result = installer.install(source, PAYLOAD, source / 'backups')
        backup = Path(result['backup'])
        for name, content in old_files.items():
            self.assertEqual((backup / name).read_bytes(), content)
        for name in old_files:
            (agent / name).write_bytes((backup / name).read_bytes())
        self.assertEqual({name: (agent / name).read_bytes() for name in old_files}, old_files)


if __name__ == '__main__':
    unittest.main()
