<?php
    require_once 'controller/service/view.php';

    class testController {
        public function start() {
            return View::createView('testpage.php', []);
        }
    }
?>