<?php
    require_once 'controller/service/view.php';

    class mainPageController {
        public function start() {
            return View::createView('mainPage.php', []);
        }
    }
?>