import contextlib
import io
import queue
import unittest
from unittest.mock import patch
import main
from robot_faq import answer
import offline_speech


class VoiceTests(unittest.TestCase):
    def setUp(self):
        main.TEXT_MODE = False
        main.robot_speaking = False
        main.audio_queue = queue.Queue(maxsize=2)
        main.recognizer_needs_reset.clear()

    def test_unknown_words_cannot_become_motion(self):
        self.assertEqual(main.classify_command('[unk] forward'), 'unknown')
        self.assertEqual(main.classify_command('do not move forward'), 'unknown')
        self.assertEqual(main.classify_command(' MOVE   FORWARD '), 'move_forward')

    def test_all_command_phrases_are_routable(self):
        for phrase in main.VOSK_COMMANDS:
            if phrase != '[unk]' and answer(phrase) is None:
                self.assertNotEqual(main.classify_command(phrase), 'unknown', phrase)

    def test_speech_audio_is_discarded_even_after_tts_failure(self):
        main.audio_queue.put(b'old command')
        def failed_speech(text):
            main.audio_callback(b'echo', 0, None, None)
            raise RuntimeError('speaker unavailable')
        with patch('main.say', side_effect=failed_speech):
            main.speak('hello')
        self.assertTrue(main.audio_queue.empty())
        self.assertFalse(main.robot_speaking)
        self.assertTrue(main.recognizer_needs_reset.is_set())

    def test_audio_overflow_drops_stale_commands(self):
        for data in [b'one', b'two', b'three']:
            main.audio_callback(data, 0, None, None)
        self.assertTrue(main.audio_queue.empty())
        self.assertTrue(main.recognizer_needs_reset.is_set())

    def test_motion_does_not_claim_hardware_success(self):
        main.TEXT_MODE = True
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main.execute_action('move_forward')
        self.assertIn('Motor controller is not connected', output.getvalue())

    def test_faq_does_not_invent_sensor_data(self):
        self.assertIn('not connected', answer('read soil moisture'))
        self.assertIsNone(answer('water the field'))

    def test_linux_tts_passes_text_as_input(self):
        with patch('offline_speech.os.name', 'posix'), patch('offline_speech.shutil.which', return_value='/usr/bin/espeak-ng'), patch('offline_speech.subprocess.run') as run:
            offline_speech.say('hello')
        self.assertEqual(run.call_args.args[0], ['/usr/bin/espeak-ng', '--stdin'])
        self.assertEqual(run.call_args.kwargs['input'], 'hello')


if __name__ == '__main__':
    unittest.main()
